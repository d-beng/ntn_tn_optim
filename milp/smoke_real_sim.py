# milp/smoke_real_sim.py

#!/usr/bin/env python3
"""
smoke_real_sim.py — ONE-HEX smoke test of the simulator-in-the-loop oracle.

Gates everything downstream (province --real-sim, coordination ladder).
  A.  cfg / region / leos built the scenario.py way (Hydra composes defaults:)
  A2. PRE-FLIGHT: geometry + coverage feasibility, BEFORE any solve
  B.  BaseStation objects from a MILP placement
  C.  run_single_hour executes exactly one step (hour 20)
  D.  user attribute names + DROP-REASON CENSUS   <- primary diagnostic
  D2. coordination gap (upper bound on what steering could recover)
  E.  both feedbacks (eta, rho) come back non-degenerate
  F.  RESCORE THE SAME PLACEMENT under corrected physics
  G.  optional drop-targeted loop

WHY THE PRE-FLIGHT EXISTS
-------------------------
A run with +agg_res=11 +agg_safety=1.0 +rho_dep=0.95 +outage_budget_mbps=0
sat in HiGHS for 4+ hours with no output. It was not slow -- it was almost
certainly INFEASIBLE, and everything needed to know that was already
computed by build_instance:

  * conflicts force deployed mid-band sites >= R*sqrt(3)*rho_dep apart, so
    the densest LEGAL network has covering radius R*rho_dep;
  * eligibility stops at r_eff = R - agg_safety*circumradius(agg_res);
  * if r_eff < R*rho_dep, coverage holes are GUARANTEED -> some demand points
    have an empty eligibility set;
  * those points can only be served by NTN, which is capped at C_beam per
    res-5 hex. If their demand exceeds that ceiling, outage_budget_mbps=0 is
    unsatisfiable by ANY deployment.

The feasibility condition is closed-form:

    agg_safety * circumradius(agg_res)  <=  R_t * (1 - rho_dep)

At rho_dep=0.95 the entire budget for UMi (R=311 m) is 15.6 m. res-11's
circumradius is 28.7 m -> misses by 13 m. rho_dep is the CHEAP side of that
inequality: lowering it costs nothing computationally, and make_real_se_fn
computes the interferer ring at ISD = R*sqrt(3)*rho_dep, so the SE stays
self-consistent automatically.

OTHER FIXES IN THIS REVISION
----------------------------
* [F] RESCORES the placement (LP with y fixed) instead of re-solving. The old
  {RMA:0,UMA:0,UMI:0} / 0.23% was a stuck time-limit incumbent, not economics.
* [G] passes lam / c_beam explicitly; it used to inherit lam=100, c_beam=5.
* The first solve now LOGS by default (+solver_log=false to silence). A
  multi-hour silent solve should not be possible by accident.
* +size_only=true prints model dimensions and exits.
* Optional chance-constrained solve via +risk_z (default 0.0 = off).

    export PYTHONPATH=/Utilisateurs/dbenguer/ntn_tn_optim/src
    python smoke_real_sim.py workers=200 +solver_threads=64 \
        +agg_res=10 +agg_safety=1.0 +rho_dep=0.75 +rho_cand=0.5 \
        +k_elig=8 +dens_uma=0 +dens_umi=0 \
        +outage_budget_mbps=0 +solve_time_limit=21600
"""
from __future__ import annotations
import copy
import json
import math
import pickle
import sys
import time
from collections import Counter

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict

from hybrid_ntn_optimizer.core.types import OrbitType, WalkerParameters
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region

sys.path.insert(0, ".")

DEF_USERS = "/Utilisateurs/dbenguer/ntn_tn_optim/data/users.pkl"


def _hav_km(a1, b1, a2, b2):
    p1, p2 = math.radians(a1), math.radians(a2)
    dp, dl = p2 - p1, math.radians(b2 - b1)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(x))


# ---------------------------------------------------------------------------
def preflight(inst, tiers, rho_dep, rho_cand, agg_res, agg_safety,
              outage_budget):
    """Everything decidable WITHOUT running the solver.

    Raises SystemExit on a certain infeasibility; prints the closed-form fix
    for a likely one. Costs milliseconds.
    """
    import h3
    from candidate_generator import band_of

    eps_km = h3.average_hexagon_edge_length(agg_res, unit="km")
    print(f"[A2] pre-flight: agg_res={agg_res} (circumradius "
          f"{1000*eps_km:.1f} m), agg_safety={agg_safety}, "
          f"rho_dep={rho_dep}, rho_cand={rho_cand}", flush=True)

    # ---- 1. covering geometry, per tier -----------------------------------
    # A triangular lattice with nearest-neighbour spacing s leaves deep holes
    # at s/sqrt(3). Conflicts force s >= R*sqrt(3)*rho_dep for mid/mmw band,
    # so the densest LEGAL network covers only to radius R*rho_dep. Low band
    # (RMa) has no conflict rows, so its limit comes from the CANDIDATE
    # lattice instead: R*rho_cand.
    holes = []
    for ti, t in enumerate(tiers):
        r_eff = inst.r_eff_km[ti] if inst.r_eff_km else t.radius_km
        if band_of(t) == "low":
            need = t.radius_km * rho_cand
            why = f"rho_cand={rho_cand}"
            knob = rho_cand
        else:
            need = t.radius_km * rho_dep
            why = f"rho_dep={rho_dep}"
            knob = rho_dep
        slack = 1000.0 * (r_eff - need)
        ok = slack >= 0.0
        print(f"      {t.name:8s} r_eff={1000*r_eff:8.1f} m  need="
              f"{1000*need:8.1f} m ({why})  slack={slack:+8.1f} m  "
              f"{'OK' if ok else 'HOLES GUARANTEED'}", flush=True)
        if not ok:
            holes.append((t, slack, knob, why.split('=')[0]))

    if holes:
        print("      " + "-" * 62, flush=True)
        for t, slack, knob, knob_name in holes:
            max_rho = 1.0 - agg_safety * eps_km / t.radius_km
            max_safety = t.radius_km * (1.0 - knob) / max(eps_km, 1e-9)
            print(f"      {t.name}: short by {-slack:.1f} m. Fix ONE of:\n"
                  f"          {knob_name:9s}  <= {max_rho:.3f}   "
                  f"(cheapest: no model growth; SE is recomputed at the "
                  f"tighter ISD, so it stays self-consistent)\n"
                  f"          agg_safety <= {max_safety:.2f}\n"
                  f"          agg_res    -> finer (costs demand points)",
                  flush=True)

    # ---- 2. how much demand is structurally unreachable --------------------
    U = len(inst.dem_mbps)
    no_elig = [u for u in range(U) if not inst.elig_j[u]]
    d_unreach = float(sum(inst.dem_mbps[u] for u in no_elig))
    ntn_cap = float(sum(inst.beam_cap_of.get(h, inst.beam_cap_mbps)
                        for h in inst.hex_ids))
    print(f"      {len(no_elig):,} of {U:,} demand points have NO eligible "
          f"candidate, carrying {d_unreach/1e3:,.2f} Gbps", flush=True)
    print(f"      total NTN ceiling over {len(inst.hex_ids)} hexes: "
          f"{ntn_cap/1e3:,.2f} Gbps", flush=True)

    if outage_budget is not None and d_unreach > ntn_cap + float(outage_budget):
        raise SystemExit(
            f"\nINFEASIBLE BEFORE SOLVING.\n"
            f"  {d_unreach/1e3:,.2f} Gbps sits at demand points that NO "
            f"terrestrial candidate can reach.\n"
            f"  NTN can absorb at most {ntn_cap/1e3:,.2f} Gbps "
            f"(one beam per res-5 hex).\n"
            f"  outage_budget_mbps={outage_budget} is therefore unsatisfiable "
            f"by ANY deployment on this candidate set.\n"
            f"  Not starting the solver -- it would spend hours proving this.\n"
            f"  Fix the geometry printed above, raise outage_budget_mbps, or "
            f"drop the budget entirely (lambda-penalty only).")

    if d_unreach > 0:
        print(f"      -> that demand can ONLY be served by NTN; it will show "
              f"up as no_eligible outage once the beams are full.", flush=True)
    return True


# ---------------------------------------------------------------------------
@hydra.main(version_base=None,
            config_path="/Utilisateurs/dbenguer/ntn_tn_optim/configs",
            config_name="base")
def main(cfg: DictConfig):
    for g in ("constellation", "scenario", "terrestrial", "simulation"):
        if g not in cfg:
            raise SystemExit(f"config group '{g}' missing after Hydra "
                             f"composition -- check configs/{g}/ and the "
                             f"defaults: list in your root config")

    workers = int(cfg.get("workers", 8))
    solver_threads = int(cfg.get("solver_threads", min(workers, 16)))
    solver_log = bool(cfg.get("solver_log", True))
    lam = float(cfg.get("lam", 1309.0))
    c_beam = float(cfg.get("c_beam", 49813.0))
    users_path = str(cfg.get("users", DEF_USERS))
    rho_dep = float(cfg.get("rho_dep", 0.95))
    rho_cand = float(cfg.get("rho_cand", 1.0))
    agg_res = int(cfg.get("agg_res", 10))
    agg_safety = float(cfg.get("agg_safety", 1.0))
    k_elig = int(cfg.get("k_elig", 8))
    risk_z = float(cfg.get("risk_z", 0.0))
    disagg = bool(cfg.get("disaggregate_linking", True))
    closest_assignment=bool(cfg.get("closest_assignment", False))
    # CROSS-TIER minimum separation in METRES. 75 m default: a macro umbrella
    # with small cells underneath is standard HetNet practice. The legacy rule
    # used the small tier's full design ISD (512 m for UMa-UMi), which exceeds
    # a UMa's own 474 m radius -- so one UMa forbade every UMi in its whole
    # footprint. Set +cross_tier_m=legacy to reproduce the old behaviour,
    # +cross_tier_m=none to remove the cross-tier constraint entirely.
    xt_m = cfg.get("cross_tier_m", 75.0)
    if isinstance(xt_m, str):
        xt_m = None if xt_m.lower() in ("none", "null", "off") else xt_m

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
    print(f"[.] loading {users_path} ...", flush=True)
    t0 = time.time()
    with open(users_path, "rb") as f:
        users = pickle.load(f)
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

    # OPTIONAL halo trim: by default the whole of ring-1 is included. Set
    # +halo_km=2.847 for the same halo BAND run_real_tile.py uses -- far
    # smaller instance, identical centre-hex physics.
    halo_km = cfg.get("halo_km", None)
    if halo_km is None:
        sel = [i for c in neigh if c in cells for i in cells[c]]
    else:
        clat, clon = h3.cell_to_latlng(hex_id)
        r_cut = h3.average_hexagon_edge_length(5, unit="km") + float(halo_km)
        sel = []
        for c in neigh:
            if c not in cells:
                continue
            if c == hex_id:
                sel.extend(cells[c])
            else:
                sel.extend(i for i in cells[c]
                           if _hav_km(users[i].current_lat,
                                      users[i].current_lon, clat, clon) <= r_cut)
    hex_users = [users[i] for i in sel]
    del users

    lat = np.array([u.current_lat for u in hex_users])
    lon = np.array([u.current_lon for u in hex_users])
    dem = np.array([u.get_demand_at_time(20.0) for u in hex_users])
    print(f"    hex {hex_id}: {len(hex_users):,} users "
          f"({'ring-1 full' if halo_km is None else f'halo {halo_km} km'}), "
          f"demand {dem.sum()/1e3:.1f} Gbps ({dem.mean():.3f} Mbps/user)",
          flush=True)

    # ---------- build the instance ----------
    from candidate_generator import build_instance
    from hex_milp import (solve_hex, solve_hex_min_outage, diagnose_outage,
                          rescore_fixed_placement)
    scen_cfg = cfg.terrestrial.scenarios
    tiers = load_tiers(scen_cfg,
                       float(cfg.get("dens_uma", 1600)),
                       float(cfg.get("dens_umi", 4000)),
                       float(cfg.get("dens_mmw", 15000)))
    se_fn = make_real_se_fn(scen_cfg, rho_dep=rho_dep)
    t0 = time.time()
    _xt_kw = ({"cross_tier_midband_conflict": True}
              if isinstance(xt_m, str) and xt_m.lower() == "legacy"
              else {"cross_tier_min_dist_km": (None if xt_m is None
                                               else float(xt_m) / 1000.0)})
    inst = build_instance(lat, lon, dem, hex_id, tiers=tiers, se_fn=se_fn,
                          rho_cand=rho_cand, rho_dep=rho_dep,
                          K_elig=k_elig, agg_res=agg_res,
                          agg_safety=agg_safety, **_xt_kw)
    n_x = sum(len(e) for e in inst.elig_j)
    print(f"    instance: {len(inst.cand_tier):,} candidates, "
          f"{len(inst.dem_mbps):,} demand pts (res-{inst.agg_res}, "
          f"agg_safety={inst.agg_safety}), {n_x:,} x-columns, "
          f"{len(inst.conflict_pairs):,} conflicts "
          f"(cross_tier={'off' if xt_m is None else xt_m})"
          f"  [{time.time()-t0:.0f}s]", flush=True)
    # CONFLICTS CAP HOW MANY SITES MAY BE OPENED AT ALL. Measured across three
    # runs on this hex: degree 1.9 -> 60% of candidates openable -> 100%
    # served; degree 5.6 -> 30% openable -> 73% served. When the degree is
    # high, no lambda and no time limit can buy capacity.
    _deg = 2 * len(inst.conflict_pairs) / max(len(inst.cand_tier), 1)
    _mis = len(inst.cand_tier) * math.log(_deg + 1) / (_deg + 1)
    print(f"    conflict graph: mean degree {_deg:.1f}; roughly {_mis:,.0f} of "
          f"{len(inst.cand_tier):,} candidates can be open simultaneously "
          f"(independent-set estimate)", flush=True)

    tl = float(cfg.get("solve_time_limit", 1800))
    _budget = cfg.get("outage_budget_mbps", None)
    _budget = None if _budget is None else float(_budget)

    # ---------- A2. PRE-FLIGHT (before any solver time is spent) ----------
    preflight(inst, tiers, rho_dep, rho_cand, agg_res, agg_safety, _budget)

    if bool(cfg.get("size_only", False)):
        print("[SIZE-ONLY] stopping before the solve as requested.", flush=True)
        return

    # ---------- solve once ----------
    if risk_z > 0:
        from risk_solve import (solve_with_demand_risk, variance_by_point,
                                sparse_sector_report)
        print(f"[.] chance-constrained solve at z={risk_z:.1f} "
              f"(margin sized per site from demand variance)", flush=True)
        res = solve_with_demand_risk(
            inst, z=risk_z, n_passes=int(cfg.get("risk_passes", 3)),
            lam=lam, c_beam=c_beam, time_limit_s=tl, threads=solver_threads,
            require_coverage=bool(cfg.get("require_coverage", False)),
            outage_budget_mbps=_budget, disaggregate_linking=disagg, log=True)
        if not res.get("no_solution") and res["y"].any():
            sparse_sector_report(inst, res, variance_by_point(inst), z=risk_z)
    else:
        res = solve_hex(inst, lam=lam, c_beam=c_beam,
                        time_limit_s=tl, threads=solver_threads,
                        log=solver_log, disaggregate_linking=disagg,
                        require_coverage=bool(cfg.get("require_coverage",
                                                      False)),
                        outage_budget_mbps=_budget,closest_assignment=bool(cfg.get("closest_assignment", False)))

    if res["status"].startswith("Infeasible") or res.get("no_solution"):
        # Do NOT abandon the run. The lexicographic form ALWAYS returns a
        # placement plus eps* = the provably minimum achievable outage.
        print(f"    !! INFEASIBLE at outage budget {_budget}. Falling back to "
              "LEXICOGRAPHIC min-outage to certify the floor. NOTE: that is "
              "TWO more solves of the same size -- if the [A2] pre-flight "
              "flagged holes, fix the geometry instead of waiting.", flush=True)
        res = solve_hex_min_outage(inst, lam=lam, c_beam=c_beam,
                                   time_limit_s=tl, threads=solver_threads,
                                   log=False)
        print(f"    CERTIFIED FLOOR: "
              f"{res.get('min_outage_mbps', res['outage_mbps']):,.0f} Mbps is "
              f"the MINIMUM outage any deployment on this candidate set can "
              f"achieve. status={res['status']}", flush=True)

    n_sites0 = int(sum(res["opened"].values()))
    print(f"[.] MILP: {res['status']} gap={res['gap']:.3f} "
          f"obj={res['objective']:,.0f} served={res['served_pct']:.2f}% "
          f"cand={len(inst.cand_tier):,} opened={res['opened']} "
          f"({n_sites0:,} sites) assoc_pairs={len(res['x_assign'])}",
          flush=True)
    if n_sites0 == 0:
        raise SystemExit(
            f"MILP opened ZERO sites. At lam={lam:,.0f} that cannot be optimal "
            f"on a populated hex -- it is a solver artifact (time limit / "
            f"trivial incumbent), not economics. Raise solve_time_limit, "
            f"loosen mip_gap, or check that elig_se is not all ~0.")

    _w = diagnose_outage(inst, res)
    print(f"    MODEL's OWN OUTAGE {res['outage_mbps']:,.0f} Mbps splits as:"
          f"  no_eligible={_w['no_eligible']/1e3:.1f}"
          f"  all_open_full={_w['elig_all_open_full']/1e3:.1f}"
          f"  conflict_blocked={_w.get('conflict_blocked', 0.0)/1e3:.1f}"
          f"  not_opened={_w['elig_not_opened']/1e3:.1f} Gbps", flush=True)
    print("      no_eligible      -> CANDIDATE SET too sparse (radius/lattice)\n"
          "      all_open_full    -> CAPACITY wall (spectrum/beam)\n"
          "      conflict_blocked -> PACKING GEOMETRY: the model was NOT "
          "ALLOWED to open more sites\n"
          "                          (rho_dep / rho_cand / cross_tier_m). "
          "Raising lambda will NOT help.\n"
          "      not_opened       -> ECONOMICS: lambda too low vs site cost",
          flush=True)
    if res["status"].startswith("Time"):
        print(f"    !! TIME LIMIT: incumbent is NOT proven optimal (gap "
              f"{res['gap']:.1%}). Raise solve_time_limit= or accept the gap.",
              flush=True)

    # ---------- B. towers ----------
    from real_sim_oracle import (build_base_stations,
                                 make_real_simulation_oracle,
                                 coordination_gap)
    bss, backmap = build_base_stations(inst, res["y"], scen_cfg)
    print(f"[B] built {len(bss)} sector cells ({len(bss)//3} sites)", flush=True)

    # ---------- C. exactly one simulated hour ----------
    from full_pipeline_hooks import run_single_hour, user_result_fields
    sim_cfg = copy.deepcopy(cfg)
    try:      # full_pipeline: range(20*3600, duration_s+step, step) -> [72000]
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
        if getattr(u, "tn_eval_bs", None):
            n_tn += 1
        elif getattr(u, "ntn_eval_beam", None):
            n_ntn += 1
        else:
            n_none += 1
    print(f"    attach census: TN={n_tn:,}  NTN={n_ntn:,}  neither={n_none:,}",
          flush=True)

    # ---- DROP-REASON CENSUS: the single most decisive diagnostic ----------
    reasons = Counter()
    short_by_reason = Counter()
    sim_drop_mbps = 0.0
    sim_demand_mbps = 0.0
    for u in hex_users:
        _, _, got, dem_u, _ = user_result_fields(u)
        sim_demand_mbps += dem_u
        if got + 1e-9 >= dem_u:
            continue
        r = str(getattr(u, "tn_reason", "?") or "?")
        reasons[r] += 1
        short_by_reason[r] += (dem_u - got)
        sim_drop_mbps += (dem_u - got)
    print("    --- WHY USERS WERE SHORT (tn_reason census) ---", flush=True)
    for r, n in reasons.most_common(10):
        print(f"      {n:>9,} users  {short_by_reason[r]/1e3:>8.1f} Gbps  {r}",
              flush=True)
    print(f"    SIM TOTAL: {sim_drop_mbps/1e3:,.1f} Gbps dropped of "
          f"{sim_demand_mbps/1e3:,.1f} Gbps offered "
          f"({100*sim_drop_mbps/max(sim_demand_mbps,1e-9):.2f}%)", flush=True)
    if abs(sim_demand_mbps - float(dem.sum())) > 0.01 * max(dem.sum(), 1.0):
        print(f"    !! DEMAND MISMATCH: hooks see {sim_demand_mbps:,.0f} Mbps, "
              f"MILP provisioned {dem.sum():,.0f} Mbps. Drops are not "
              f"comparable to model outage until this is reconciled.",
              flush=True)
    print("    ^ 'No 5G Tower in Geographic Range' => COVERAGE problem "
          "(rho_dep / agg_safety / rho_cand -- see the [A2] pre-flight).\n"
          "      '5G Congestion' / 'Tower Empty'    => ASSOCIATION problem "
          "(steering/CIO), NOT solved by more sites.", flush=True)

    # ---------- D2. COORDINATION GAP ---------
    coordination_gap(hex_users, bss, log=True)

    # ---------- E. feedbacks ----------
    oracle = make_real_simulation_oracle(inst, cfg, hex_users, leos, region,
                                         scen_cfg, workers=workers,
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

    # ---------- F. RESCORE THE SAME PLACEMENT under corrected physics -------
    # NOT a re-solve. Freezing y turns (8) into an LP over (x,z,o) plus the
    # beam integers, so this cannot return a stuck incumbent. It answers the
    # only question that matters here: does the [C] network still deliver once
    # SE is the simulator's rather than the planner's?
    inst_corr = copy.deepcopy(inst)
    for u in range(len(inst_corr.dem_mbps)):
        inst_corr.elig_se[u] = [0.5 * a + 0.5 * b
                                for a, b in zip(inst.elig_se[u], realized[u])]
    cap_F = rho or None
    if risk_z > 0 and res.get("risk_cap"):
        from risk_solve import merge_cap
        cap_F = merge_cap(res.get("risk_cap"), rho)
    res2 = rescore_fixed_placement(inst_corr, res["y"], lam=lam, c_beam=c_beam,
                                   cap_factor=cap_F,
                                   time_limit_s=min(tl, 600.0),
                                   threads=solver_threads, log=False)
    print(f"[F] RESCORED the same {res2['rescored_sites']:,} sites under "
          f"corrected SE/rho: {res2['status']} gap={res2['gap']:.4f}  "
          f"served={res2['served_pct']:.2f}%  "
          f"model_outage={res2['outage_mbps']:,.0f} Mbps", flush=True)
    print(f"    planner said {res['served_pct']:.2f}% served | corrected model "
          f"says {res2['served_pct']:.2f}% | simulator measured "
          f"{100*(1-sim_drop_mbps/max(sim_demand_mbps,1e-9)):.2f}%", flush=True)
    print("    ^ if the last two agree, the correction loop is validated on "
          "this instance -- that IS the result, not a bug.", flush=True)

    # ---------- G. OPTIONAL: drive the SIMULATOR's drops to a target -------
    tgt = cfg.get("drop_target", None)
    if tgt is not None:
        from drop_targeted_loop import solve_to_drop_target
        print(f"\n[G] drop-targeted loop: target {float(tgt):,.0f} Mbps, "
              f"{int(cfg.get('drop_iters', 3))} iterations max "
              f"(lam={lam:,.0f}, c_beam={c_beam:,.0f})", flush=True)
        print("    NOTE: this loop RE-CHOOSES the placement every iteration. "
              "If the census above is coverage/association dominated, expect "
              "it to diverge and stop -- that is the loop working, not "
              "failing.", flush=True)
        resG, histG = solve_to_drop_target(
            inst, oracle, target_drop_mbps=float(tgt),
            max_iter=int(cfg.get("drop_iters", 3)),
            beta=float(cfg.get("drop_beta", 1.0)),
            require_coverage=bool(cfg.get("require_coverage", False)),
            lam=lam, c_beam=c_beam,
            time_limit_s=tl, threads=solver_threads, log=True)
        print("[G] trajectory (sim drops, Mbps):",
              [round(h["sim_drop_mbps"]) for h in histG], flush=True)
        print("[G] sites:", [h["sites"] for h in histG], flush=True)
        print(f"[G] baseline from [C] was {sim_drop_mbps:,.0f} Mbps at "
              f"{n_sites0:,} sites -- if no iteration beats that, keep [C].",
              flush=True)

    print("\nSMOKE TEST COMPLETE" if rho else
          "\nSMOKE INCOMPLETE: fix user_result_fields per [D], rerun")


if __name__ == "__main__":
    main()