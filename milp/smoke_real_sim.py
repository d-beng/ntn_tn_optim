#!/usr/bin/env python3
"""
smoke_real_sim.py — ONE-HEX smoke test of the simulator-in-the-loop oracle.

Gates everything downstream (province --real-sim, coordination ladder).
  A. cfg / region / leos built the scenario.py way (Hydra)
  B. BaseStation objects from a MILP placement
  C. run_single_hour executes exactly one step (hour 20)
  D. user attribute names after simulation   <- __slots__-safe dump
  E. both feedbacks (eta, rho) come back non-degenerate
  F. one corrected re-solve

    export PYTHONPATH=/Utilisateurs/dbenguer/ntn_tn_optim/src
    python smoke_real_sim.py hex=852b9bc7fffffff workers=32
"""
from __future__ import annotations
import copy
import json
import pickle
import sys
import time

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict

from hybrid_ntn_optimizer.core.types import OrbitType, WalkerParameters
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region

sys.path.insert(0, ".")


@hydra.main(version_base=None,
            config_path="/Utilisateurs/dbenguer/ntn_tn_optim/configs",
            config_name="base")
def main(cfg: DictConfig):
    # ---------- A. config + world objects ----------
    print("[A] building cfg / region / leos ...", flush=True)
    from hybrid_ntn_optimizer.models.scenario import Region
    from hybrid_ntn_optimizer.constellation.leo import LEOConstellation

    with open(to_absolute_path(cfg.scenario.geojson_path)) as f:
        geometry = json.load(f)
    region = Region(name=cfg.scenario.name, geojson_geometry=geometry,
                    h3_resolution=cfg.scenario.h3_resolution)
    tessellate_region(region, pad_edges=True)

    leos, total_sats = [], 0
    for shell_key, shell_cfg in cfg.constellation.shells.items():
        wp = WalkerParameters(
            total_satellites=shell_cfg.total_satellites,
            num_planes=shell_cfg.num_planes, phasing=shell_cfg.phasing,
            inclination_deg=shell_cfg.inclination_deg,
            altitude_km=shell_cfg.altitude_km, orbit_type=OrbitType.LEO)
        shell = LEOConstellation(
            params=wp, name=shell_cfg.name,
            eirp_dbw=cfg.constellation.get("eirp_dbw", 40.0),
            g_t_db=cfg.constellation.get("g_t_db", 10.0),
            max_spot_beams=cfg.constellation.get("max_spot_beams", 15),
            beam_radius_nadir_km=cfg.constellation.get("beam_radius_nadir_km", 200.0),
            max_steering_angle_deg=cfg.constellation.get("max_steering_angle_deg", 45.0))
        leos.append(shell)
        total_sats += shell.num_satellites
    print(f"    region tessellated; {total_sats} sats in {len(leos)} shells",
          flush=True)

    # ---------- load users, pick hex ----------
    import h3
    from run_real_tile import load_tiers, make_real_se_fn
    print(f"[.] loading {cfg.users} ...", flush=True)
    t0 = time.time()
    users = pickle.load(open(cfg.users, "rb"))
    print(f"    {len(users):,} users in {time.time()-t0:.0f}s", flush=True)
    print("[.] moving users to hour 20 ...", flush=True)
    for u in users:
        u.move(20.0, 5)

    cells = {}
    for i, u in enumerate(users):
        cells.setdefault(h3.latlng_to_cell(u.current_lat, u.current_lon, 5),
                         []).append(i)
    hex_id = cfg.get("hex") or max(cells, key=lambda c: len(cells[c]))
    neigh = set(h3.grid_disk(hex_id, 1))
    sel = [i for c in neigh if c in cells for i in cells[c]]
    hex_users = [users[i] for i in sel]
    del users

    lat = np.array([u.current_lat for u in hex_users])
    lon = np.array([u.current_lon for u in hex_users])
    dem = np.array([u.get_demand_at_time(20.0) for u in hex_users])
    print(f"    hex {hex_id}: {len(hex_users):,} users (with ring-1), "
          f"demand {dem.sum()/1e3:.1f} Gbps "
          f"({dem.mean():.3f} Mbps/user)", flush=True)

    # ---------- solve once ----------
    from candidate_generator import build_instance
    from hex_milp import solve_hex
    scen_cfg = cfg.terrestrial.scenarios
    tiers = load_tiers(scen_cfg,
                       float(cfg.get("dens_uma", 1600)),
                       float(cfg.get("dens_umi", 4000)),
                       float(cfg.get("dens_mmw", 15000)))
    se_fn = make_real_se_fn(scen_cfg)
    inst = build_instance(lat, lon, dem, hex_id, tiers=tiers, se_fn=se_fn,
                          rho_cand=float(cfg.get('rho_cand', 1.0)),
                          rho_dep=float(cfg.get('rho_dep', 0.95)),
                          K_elig=int(cfg.get('k_elig', 8)),
                          agg_res=int(cfg.get('agg_res', 9)),
                          agg_safety=float(cfg.get('agg_safety', 1.0)))
    tl = float(cfg.get("solve_time_limit", 1800))
    _budget = cfg.get("outage_budget_mbps", None)
    res = solve_hex(inst, time_limit_s=tl, threads=cfg.workers, log=False,
                    require_coverage=bool(cfg.get("require_coverage", False)),
                    outage_budget_mbps=(None if _budget is None
                                        else float(_budget)))
    if res["status"].startswith("Infeasible"):
        # Do NOT abandon the run. Switch to the lexicographic form, which
        # ALWAYS returns a placement plus eps* = the provably minimum outage
        # achievable on this candidate set. eps* is the certificate: "no
        # deployment on J does better than this".
        from hex_milp import solve_hex_min_outage
        print(f"    !! INFEASIBLE at outage budget {_budget}. Falling back to "
              "LEXICOGRAPHIC min-outage to certify the floor.", flush=True)
        res = solve_hex_min_outage(inst, time_limit_s=tl, threads=cfg.workers,
                                   log=False)
        print(f"    CERTIFIED FLOOR: {res.get('min_outage_mbps', res['outage_mbps']):,.0f}"
              f" Mbps is the MINIMUM outage any deployment on this candidate "
              f"set can achieve. status={res['status']}", flush=True)
    print(f"[.] MILP: {res['status']} gap={res['gap']:.3f} "
          f"served={res['served_pct']:.2f}% cand={len(inst.cand_tier):,} "
          f"opened={res['opened']} assoc_pairs={len(res['x_assign'])}",
          flush=True)
    from hex_milp import diagnose_outage
    _w = diagnose_outage(inst, res)
    print(f"    MODEL's OWN OUTAGE {res['outage_mbps']:,.0f} Mbps splits as:"
          f"  no_eligible={_w['no_eligible']/1e3:.1f}"
          f"  all_open_full={_w['elig_all_open_full']/1e3:.1f}"
          f"  not_opened={_w['elig_not_opened']/1e3:.1f} Gbps", flush=True)
    print("      no_eligible   -> CANDIDATE SET too sparse (radius/lattice)\n"
          "      all_open_full -> CAPACITY wall (spectrum/beam)\n"
          "      not_opened    -> ECONOMICS: lambda too low vs site cost",
          flush=True)
    if res["status"].startswith("Time"):
        print(f"    !! TIME LIMIT: incumbent is NOT proven optimal (gap "
              f"{res['gap']:.1%}). Raise solve_time_limit= or accept the gap.",
              flush=True)

    # ---------- B. towers ----------
    from real_sim_oracle import build_base_stations, make_real_simulation_oracle
    bss, backmap = build_base_stations(inst, res["y"], scen_cfg)
    print(f"[B] built {len(bss)} sector cells ({len(bss)//3} sites)", flush=True)

    # ---------- C. exactly one simulated hour ----------
    from full_pipeline_hooks import run_single_hour
    sim_cfg = copy.deepcopy(cfg)
    try:                       # full_pipeline: range(20*3600, duration+step, step)
        with open_dict(sim_cfg):
            sim_cfg.simulation.duration_s = 72000
            sim_cfg.simulation.time_step_s = 3600
    except Exception as e:
        print(f"    (could not force single-hour cfg: {e})", flush=True)
    t0 = time.time()
    run_single_hour(sim_cfg, hex_users, bss, leos, region)
    print(f"[C] single-hour simulation done in {time.time()-t0:.0f}s", flush=True)

    # ---------- D. attribute discovery (__slots__-safe) ----------
    u0 = hex_users[0]
    names = []
    for cls in type(u0).__mro__:
        names += list(getattr(cls, "__slots__", []) or [])
    if not names:
        names = [a for a in dir(u0) if not a.startswith("_")]
    names = sorted(set(names))
    print("[D] User attributes after simulation (first user):", flush=True)
    for a in names:
        try:
            v = getattr(u0, a)
        except Exception:
            continue
        if callable(v):
            continue
        if isinstance(v, (int, float, str, bool, type(None))):
            print(f"      {a} = {v}")
    n_tn = n_ntn = n_none = 0
    for u in hex_users:
        bs = getattr(u, "tn_eval_bs", None)
        if bs:
            n_tn += 1
        elif getattr(u, "ntn_eval_beam", None):
            n_ntn += 1
        else:
            n_none += 1
    print(f"    attach census: TN={n_tn:,}  NTN={n_ntn:,}  neither={n_none:,}",
          flush=True)
    for u in hex_users:                 # must be TN-ATTACHED to be informative
        if getattr(u, "tn_eval_bs", None):
            print("    --- a TN-ATTACHED user (non-zero fields only) ---",
                  flush=True)
            for a in names:
                try:
                    v = getattr(u, a)
                except Exception:
                    continue
                if isinstance(v, (int, float, str, bool)) and v not in (0, False, ""):
                    print(f"      {a} = {v}")
            break
    # ---- DROP-REASON CENSUS: the single most decisive diagnostic ----------
    from collections import Counter
    reasons = Counter()
    short_by_reason = Counter()
    for u in hex_users:
        got = float(getattr(u, "served_mbps", 0.0) or 0.0)
        dem = float(getattr(u, "current_demand", 0.0) or 0.0)
        if got + 1e-9 >= dem:
            continue                      # fully served, not a drop
        r = str(getattr(u, "tn_reason", "?") or "?")
        reasons[r] += 1
        short_by_reason[r] += (dem - got)
    print("    --- WHY USERS WERE SHORT (tn_reason census) ---", flush=True)
    for r, n in reasons.most_common(10):
        print(f"      {n:>9,} users  {short_by_reason[r]/1e3:>8.1f} Gbps  {r}",
              flush=True)
    print("    ^ 'No 5G Tower in Geographic Range' => COVERAGE problem "
          "(fix: require_coverage=True, gates off).\n"
          "      '5G Congestion' / 'Tower Empty'    => CAPACITY/ASSOCIATION "
          "problem (fix: cap_factor, steering, more spectrum).", flush=True)

    print("    ^ CHECK: which attribute holds attached cell id / spec eff / "
          "served Mbps? Set them in full_pipeline_hooks.user_result_fields "
          "(ONE place).", flush=True)

    # ---------- D2. COORDINATION GAP (no simulator patch needed) ---------
    from real_sim_oracle import coordination_gap
    coordination_gap(hex_users, bss, log=True)

    # ---------- E. feedbacks ----------
    oracle = make_real_simulation_oracle(inst, cfg, hex_users, leos, region,
                                         scen_cfg, workers=cfg.workers,
                                         se_percentile=float(
                                             cfg.get("se_percentile", 20.0)),
                                         calibrate=bool(
                                             cfg.get("calibrate", True)),
                                         skip_first_sim=True,   # [C] already ran
                                         log=True)
    realized, rho, drop_by_dem = oracle(res["y"], res["x_assign"])
    changed = sum(1 for u in range(len(inst.dem_mbps))
                  for a, b in zip(inst.elig_se[u], realized[u])
                  if abs(a - b) > 1e-6)
    if rho:
        print(f"[E] eta updated on {changed:,} pairs; rho on {len(rho)} sites "
              f"(mean {np.mean(list(rho.values())):.3f})", flush=True)
    else:
        print(f"[E] eta updated on {changed:,} pairs; rho EMPTY -> the "
              "attribute names in user_result_fields are wrong, see [D]",
              flush=True)

    # ---------- F. one corrected re-solve ----------
    for u in range(len(inst.dem_mbps)):
        inst.elig_se[u] = [0.5*a + 0.5*b
                           for a, b in zip(inst.elig_se[u], realized[u])]
    res2 = solve_hex(inst, time_limit_s=tl, threads=cfg.workers, log=False,
                     cap_factor=rho or None,
                     require_coverage=bool(cfg.get("require_coverage", False)))
    print(f"[F] corrected solve: served={res2['served_pct']:.2f}% "
          f"opened={res2['opened']}  (was {res['opened']})", flush=True)

    # ---------- G. OPTIONAL: drive the SIMULATOR's drops to a target -------
    # usage:  python smoke_real_sim.py ... +drop_target=1400 +drop_iters=3
    # Each iteration = one solve + one ~22-min simulation. The loop raises
    # provisioning margin ONLY at demand points where the simulator lost
    # users (census showed congestion, not coverage, dominates).
    tgt = cfg.get("drop_target", None)
    if tgt is not None:
        from drop_targeted_loop import solve_to_drop_target
        print(f"\n[G] drop-targeted loop: target {float(tgt):,.0f} Mbps, "
              f"{int(cfg.get('drop_iters', 3))} iterations max", flush=True)
        resG, histG = solve_to_drop_target(
            inst, oracle, target_drop_mbps=float(tgt),
            max_iter=int(cfg.get("drop_iters", 3)),
            beta=float(cfg.get("drop_beta", 1.0)),
            require_coverage=bool(cfg.get("require_coverage", False)),
            time_limit_s=tl, threads=cfg.workers, log=True)
        print("[G] trajectory (sim drops, Mbps):",
              [round(h["sim_drop_mbps"]) for h in histG], flush=True)
        print("[G] sites:", [h["sites"] for h in histG], flush=True)

    print("\nSMOKE TEST COMPLETE" if rho else
          "\nSMOKE INCOMPLETE: fix user_result_fields per [D], rerun")


if __name__ == "__main__":
    main()