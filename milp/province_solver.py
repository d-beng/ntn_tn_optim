#!/usr/bin/env python3
"""
province_solver.py — multi-hex decomposition driver.

PASS 1  every dense hex solved independently (parallel across cores):
        hex + halo users, own candidates only -> placement + per-sector usage.
PASS 2  Jacobi sweep: each hex re-solved with neighbours' PASS-1 border towers
        visible as FIXED-OPEN, ZERO-COST candidates carrying their RESIDUAL
        per-sector capacity -> border overbuild pruned, edge users spill onto
        neighbours' spare spectrum. (One sweep suffices: coupling band is
        ~R_max wide vs a 16 km hex.)
STITCH  owned opened sites from the final pass -> province placement CSV.

THREE HARD BUGS FIXED IN THIS REVISION
--------------------------------------
1. `--max-outage-mbps` was declared TWICE in argparse. Python raises
   argparse.ArgumentError at startup -- this script could not run at all.
2. The sparse-hex accounting referenced `final` before it was assigned
   (NameError), and was PRESENT TWICE with slightly different logic, so the
   totals double-counted sparse demand whichever branch survived.
3. The owned-demand accounting block in _solve_one was duplicated verbatim;
   the second copy silently overwrote the first. Harmless but confusing.

Usage (cluster):
    export PYTHONPATH=/Utilisateurs/dbenguer/ntn_tn_optim/src
    python province_solver.py --users .../users.pkl \\
        --config-dir /Utilisateurs/dbenguer/ntn_tn_optim/configs \\
        --config-name base --workers 32 --min-users 500 --max-hexes 5
On the first run use --max-hexes 5 to validate the loop before the full set.
"""
from __future__ import annotations
import argparse, csv, math, os, pickle, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import h3

from candidate_generator import build_instance, unproject
from hex_milp import (solve_hex, solve_hex_min_outage, sector_usage_mhz,
                      diagnose_outage, LAM_DEFAULT, C_BEAM_DEFAULT)

# globals for fork-shared arrays (same CoW pattern as the simulator fix)
_LAT = _LON = _DEM = None
_HEX_IDX = None            # hex_id -> np.array of user indices (built ONCE)
_TIERS = _SE_FN = _SCEN = None
_ARGS = None
_ORACLE_KIND = None        # "real" | "proxy" | None (decided once per worker)


def _get_oracle(inst):
    """Physics oracle for the iterative loop. Prefers the REAL link budget
    (sim_oracle -> calculate_tn_sinr_capacity against the ACTUALLY opened
    interferer set); falls back to the analytic proxy if the package is not
    importable (local testing)."""
    global _ORACLE_KIND
    if _ORACLE_KIND != "proxy":
        try:
            from sim_oracle import make_sim_oracle
            o = make_sim_oracle(inst, _SCEN)
            _ORACLE_KIND = "real"
            return o
        except Exception as e:
            if _ORACLE_KIND is None:
                print(f"  [oracle] real physics unavailable ({e}); "
                      f"using analytic proxy", flush=True)
            _ORACLE_KIND = "proxy"
    from iterative_milp import default_oracle_factory
    return default_oracle_factory(inst, _SE_FN)


def _hav_km(a1, b1, a2, b2):
    p1, p2 = math.radians(a1), math.radians(a2)
    dp, dl = p2 - p1, math.radians(b2 - b1)
    x = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * 6371.0088 * math.asin(math.sqrt(x))


def _extract(hex_id):
    """hex + halo users via the precomputed index map (no full scans),
    vectorized haversine for the halo band."""
    clat, clon = h3.cell_to_latlng(hex_id)
    edge = h3.average_hexagon_edge_length(5, unit="km")
    r_cut = edge + _ARGS.halo_km
    parts = [_HEX_IDX[hex_id]] if hex_id in _HEX_IDX else []
    for nb in h3.grid_disk(hex_id, 1):
        if nb == hex_id or nb not in _HEX_IDX:
            continue
        idx = _HEX_IDX[nb]
        la, lo = np.radians(_LAT[idx]), np.radians(_LON[idx])
        p1 = math.radians(clat)
        dp = la - p1
        dl = lo - math.radians(clon)
        a = np.sin(dp/2)**2 + math.cos(p1)*np.cos(la)*np.sin(dl/2)**2
        d = 2*6371.0088*np.arcsin(np.sqrt(a))
        parts.append(idx[d <= r_cut])
    idx = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return _LAT[idx], _LON[idx], _DEM[idx]


def _solve_one(task):
    """PASS-1/2 solve. task = (hex_id, ext_sites|None, beam_residual|None)."""
    hex_id, ext, beam_res = task
    lat, lon, dem = _extract(hex_id)
    m = dem >= 0.1
    if m.sum() < _ARGS.min_users:
        return hex_id, None
    inst = build_instance(lat[m], lon[m], dem[m], hex_id,
                          tiers=_TIERS, se_fn=_SE_FN,
                          ext_sites=ext, beam_residual=beam_res,
                          rho_cand=_ARGS.rho_cand, rho_dep=_ARGS.rho_dep,
                          agg_res=_ARGS.agg_res, agg_safety=_ARGS.agg_safety,
                          K_elig=_ARGS.k_elig,
                          beam_bw_hz=_ARGS.beam_bw, beam_se=_ARGS.beam_se)

    use_min_out = bool(getattr(_ARGS, "min_outage", False))
    if use_min_out:
        solver_fn, skw = solve_hex_min_outage, {}
    else:
        solver_fn = solve_hex
        skw = {}
        if getattr(_ARGS, "max_outage_mbps", None) is not None:
            skw["outage_budget_mbps"] = float(_ARGS.max_outage_mbps)
        if _ARGS.zero_outage:
            skw["max_outage_frac"] = 0.0

    if getattr(_ARGS, "iterate", 1) > 1:
        # FULL LOOP: solve -> physics oracle -> correct eta -> re-solve
        from iterative_milp import solve_iterative
        res, hist = solve_iterative(
            inst, _get_oracle(inst), lam=_ARGS.lam, c_beam=_ARGS.c_beam,
            max_iter=_ARGS.iterate, damping=_ARGS.damping,
            stab_tol=_ARGS.stab_tol, mip_gap=_ARGS.gap,
            time_limit_s=_ARGS.time_limit, threads=1,
            solver_fn=solver_fn, solve_kwargs=skw, log=False)
    else:
        res = solver_fn(inst, lam=_ARGS.lam, c_beam=_ARGS.c_beam,
                        mip_gap=_ARGS.gap, time_limit_s=_ARGS.time_limit,
                        threads=1, log=False, **skw)

    if res.get("no_solution"):
        return hex_id, {"opened": [], "summary": {
            "status": res["status"], "gap": float("nan"), "objective": float("nan"),
            "served_pct": 0.0, "beams": 0, "served_tn_mbps": 0.0,
            "ntn_mbps": 0.0, "outage_mbps": res["outage_mbps"],
            "wall_s": 0.0, "opened_n": 0,
            "own_demand_mbps": float(inst.dem_mbps.sum()), "own_tn_mbps": 0.0,
            "own_ntn_mbps": 0.0, "own_out_mbps": float(inst.dem_mbps.sum()),
            "own_served_pct": 0.0, "n_own_dem_pts": 0, "own_beam": 0,
            "halo_ntn_mbps": 0.0,
            "why": {"no_eligible": 0.0, "elig_all_open_full": 0.0,
                    "elig_not_opened": 0.0, "n_pts": 0},
            "n_cand": int(len(inst.cand_tier)), "n_dem": int(len(inst.dem_mbps)),
            "tiers": {}}, "ntn_by_hex": {}}

    usage = sector_usage_mhz(inst, res)
    y = res["y"]
    opened = []
    for j in np.where(y)[0]:
        if inst.fixed_open[j]:
            continue                        # neighbour's site, not ours
        if inst.cand_owner_hex[j] != hex_id:
            continue                        # halo candidate owned elsewhere
        la, lo = unproject(inst.cand_xy[j, 0:1], inst.cand_xy[j, 1:2],
                           inst.lat0)
        W = inst.tiers[inst.cand_tier[j]].bw_hz / 1e6
        used = usage.get(j, [0.0, 0.0, 0.0])
        opened.append({
            "lat": float(la[0]), "lon": float(lo[0]),
            "tier_name": inst.tiers[inst.cand_tier[j]].name,
            "used_mhz": [round(u, 3) for u in used],
            "residual_mhz": [round(max(W - u, 0.0), 3) for u in used],
        })
    summary = {k: res[k] for k in
               ["status", "gap", "objective", "served_pct", "beams",
                "served_tn_mbps", "ntn_mbps", "outage_mbps", "wall_s"]}
    summary["opened_n"] = len(opened)

    # ---- OWNED-DEMAND accounting: the ONLY figures that may be summed -----
    # Each hex model also contains HALO users from ring-1 neighbours; those
    # users are solved again (and served) in their own hex's model. Summing a
    # model's totals counts border users up to 7x, and reports halo users as
    # "unserved" here whenever the neighbour's beam residual is exhausted --
    # even though their owner hex serves them. (Written ONCE; the old file
    # had this block twice.)
    own_u = [u for u in range(len(inst.dem_mbps)) if inst.dem_hex[u] == hex_id]
    own_set = set(own_u)
    own_d = float(sum(inst.dem_mbps[u] for u in own_u))
    own_tn = 0.0
    for (u, j), col in res["x_index"].items():
        if u in own_set:
            own_tn += float(res["x_val"][col]) * float(inst.dem_mbps[u])
    own_ntn = float(sum(res["z"][u] * inst.dem_mbps[u] for u in own_u))
    own_out = float(sum(res["o"][u] * inst.dem_mbps[u] for u in own_u))
    summary.update({
        "own_demand_mbps": own_d,
        "own_tn_mbps": own_tn,
        "own_ntn_mbps": own_ntn,
        "own_out_mbps": own_out,
        "own_served_pct": 100.0 * (own_tn + own_ntn) / max(own_d, 1e-9),
        "n_own_dem_pts": len(own_u),
        # OWNED beam accounting: this solve owns ONLY its centre hex's beam.
        # Halo users' offload onto neighbours' beams is reported separately
        # and never summed at stitch time.
        "own_beam": 1 if own_ntn > 1e-9 else 0,
        "halo_ntn_mbps": float(sum(v for k, v in res["ntn_by_hex"].items()
                                   if k != hex_id)),
    })
    summary["why"] = diagnose_outage(inst, res)
    if "history" in res:
        h0, hN = res["history"][0], res["history"][-1]
        summary["iters"] = len(res["history"])
        summary["se_drift"] = hN.get("mean_se", 0.0) - h0.get("mean_se", 0.0)
        summary["served_drift"] = hN.get("served_pct", 0.0) - h0.get("served_pct", 0.0)
        summary["oracle"] = _ORACLE_KIND
    if "min_outage_mbps" in res:
        summary["min_outage_mbps"] = res["min_outage_mbps"]
    summary["n_cand"] = int(len(inst.cand_tier))
    summary["n_dem"] = int(len(inst.dem_mbps))
    summary["tiers"] = {inst.tiers[t].name: int((inst.cand_tier == t).sum())
                        for t in set(inst.cand_tier)}
    return hex_id, {"opened": opened, "summary": summary,
                    "ntn_by_hex": res["ntn_by_hex"]}


def main():
    global _LAT, _LON, _DEM, _HEX_IDX, _TIERS, _SE_FN, _SCEN, _ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="/Utilisateurs/dbenguer/ntn_tn_optim/data/users.pkl")
    # Hydra composition: the root config is a defaults: list, so OmegaConf.load
    # on it would NOT resolve the groups. Compose it properly.
    ap.add_argument("--config-dir",
                    default="/Utilisateurs/dbenguer/ntn_tn_optim/configs")
    ap.add_argument("--config-name", default="base")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-users", type=int, default=500)
    ap.add_argument("--max-hexes", type=int, default=0, help="0 = all")
    ap.add_argument("--hour", type=float, default=20.0)
    ap.add_argument("--halo-km", type=float, default=2.847)
    ap.add_argument("--rho-cand", type=float, default=1.0)
    ap.add_argument("--rho-dep", type=float, default=0.95)
    ap.add_argument("--agg-res", type=int, default=10)
    ap.add_argument("--agg-safety", type=float, default=1.0)
    ap.add_argument("--k-elig", type=int, default=6)
    ap.add_argument("--dens-uma", type=float, default=1600.0)
    ap.add_argument("--dens-umi", type=float, default=4000.0)
    ap.add_argument("--dens-mmw", type=float, default=15000.0)
    ap.add_argument("--lam", type=float, default=LAM_DEFAULT)
    ap.add_argument("--c-beam", type=float, default=C_BEAM_DEFAULT)
    ap.add_argument("--gap", type=float, default=0.02)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--beam-bw", type=float, default=300e6)
    ap.add_argument("--beam-se", type=float, default=1.77)
    # DECLARED ONCE. The old file declared this twice -> argparse.ArgumentError
    # on import, i.e. the script could not start.
    ap.add_argument("--max-outage-mbps", type=float, default=None,
                    help="HARD cap on total outage per hex [Mbps] "
                         "(eps-constraint, Eq. 8b). e.g. 10 for near-zero. "
                         "Infeasible => that hex cannot meet the target.")
    ap.add_argument("--iterate", type=int, default=1,
                    help="physics-corrected iterations per hex (1 = single "
                         "shot; 3-5 for the full loop)")
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--stab-tol", type=float, default=0.02)
    ap.add_argument("--min-outage", action="store_true",
                    help="lexicographic: minimise outage first, then cost "
                         "(always returns a placement + the outage floor)")
    ap.add_argument("--zero-outage", action="store_true",
                    help="forbid outage entirely; INFEASIBLE proves saturation")
    ap.add_argument("--out", default="province_placement.csv")
    _ARGS = ap.parse_args()

    # ---- config: compose the Hydra defaults: list (NOT OmegaConf.load) ----
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=_ARGS.config_dir, version_base=None):
        _cfg = compose(config_name=_ARGS.config_name)
    _scen = _cfg.terrestrial.scenarios

    from run_real_tile import load_tiers, make_real_se_fn
    _TIERS = load_tiers(_scen, _ARGS.dens_uma, _ARGS.dens_umi, _ARGS.dens_mmw)
    _SE_FN = make_real_se_fn(_scen, rho_dep=_ARGS.rho_dep)
    _SCEN = _scen

    print(f"loading {_ARGS.users} ...", flush=True)
    t0 = time.time()
    with open(_ARGS.users, "rb") as f:
        users = pickle.load(f)
    print(f"  moving users to hour {_ARGS.hour} ...", flush=True)
    for u in users:
        u.move(_ARGS.hour, 5)
    _LAT = np.array([u.current_lat for u in users])
    _LON = np.array([u.current_lon for u in users])
    _DEM = np.array([u.get_demand_at_time(_ARGS.hour) for u in users])
    del users
    print(f"  {len(_LAT):,} users in {time.time()-t0:.0f}s", flush=True)

    print("  indexing users by res-5 hex (one pass)...", flush=True)
    cells = [h3.latlng_to_cell(float(a), float(b), 5)
             for a, b in zip(_LAT, _LON)]
    by_hex = defaultdict(list)
    for i, c in enumerate(cells):
        by_hex[c].append(i)
    _HEX_IDX = {c: np.array(ix, dtype=np.int64) for c, ix in by_hex.items()}
    counts = {c: len(ix) for c, ix in _HEX_IDX.items()}
    del cells, by_hex
    hexes = [hx for hx, n in sorted(counts.items(), key=lambda t: -t[1])
             if n >= _ARGS.min_users]
    if _ARGS.max_hexes:
        hexes = hexes[:_ARGS.max_hexes]
    print(f"  {len(hexes)} hexes with >= {_ARGS.min_users} users", flush=True)

    # warm the SE cache once in the parent (fork-shared afterwards)
    for t in _TIERS:
        for dkm in np.linspace(0.02, t.radius_km, 30):
            _SE_FN(float(dkm), t)

    # ---------------- PASS 1 ----------------
    import gc
    gc.collect(); gc.freeze()      # CoW protection (same fix as the simulator)
    print("\n=== PASS 1 (independent hexes) ===", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=_ARGS.workers) as ex:
        for hx, out in ex.map(_solve_one,
                              [(hx, None, None) for hx in hexes]):
            if out is None:
                continue
            results[hx] = out
            s = out["summary"]
            print(f"  {hx}  [{s['status']}]  OWN served={s['own_served_pct']:6.2f}% "
                  f"of {s['own_demand_mbps']/1e3:7.1f} Gbps  "
                  f"sites={s['opened_n']:4d}  own_beam={s['own_beam']} "
                  f"({s['own_ntn_mbps']:.0f} Mbps)  gap={s['gap']:.3f}  "
                  f"{s['wall_s']:.0f}s", flush=True)
            if "iters" in s:
                print(f"      loop[{s.get('oracle','?')}]: {s['iters']} iters  "
                      f"meanSE {s['se_drift']:+.2f}  served {s['served_drift']:+.2f} pp",
                      flush=True)
            w = s["why"]
            if w["n_pts"]:
                print(f"      cand={s['n_cand']} {s['tiers']}  dem_pts={s['n_dem']}"
                      f"  | OUTAGE Gbps: no_eligible={w['no_eligible']/1e3:.1f} "
                      f"all_open_full={w['elig_all_open_full']/1e3:.1f} "
                      f"not_opened={w['elig_not_opened']/1e3:.1f}", flush=True)

    if not results:
        raise SystemExit("PASS 1 produced no solved hexes -- lower --min-users")

    # ---------------- PASS 2 (Jacobi border sweep) ----------------
    print("\n=== PASS 2 (residual-capacity border sweep) ===", flush=True)
    edge = h3.average_hexagon_edge_length(5, unit="km")

    def border_ext(hex_id):
        """neighbours' PASS-1 opened sites within halo of this hex."""
        clat, clon = h3.cell_to_latlng(hex_id)
        ext = []
        for nb in h3.grid_disk(hex_id, 1):
            if nb == hex_id or nb not in results:
                continue
            for site in results[nb]["opened"]:
                if _hav_km(site["lat"], site["lon"], clat, clon) \
                        <= edge + _ARGS.halo_km:
                    ext.append(site)
        return ext or None

    # beam residual for FOREIGN hexes: full capacity minus what that hex's OWN
    # pass-1 solve already offloaded onto its beam. Prevents the same physical
    # beam being spent twice (once by its owner, once by a halo neighbour).
    cap_full = _ARGS.beam_bw * _ARGS.beam_se / 1e6

    def beam_res_for(hex_id):
        res_map = {}
        for nb in h3.grid_disk(hex_id, 1):
            if nb == hex_id:
                continue
            used = results[nb]["summary"]["own_ntn_mbps"] if nb in results else 0.0
            res_map[nb] = max(cap_full - used, 0.0)
        return res_map

    tasks = [(hx, border_ext(hx), beam_res_for(hx)) for hx in results]
    results2 = {}
    with ProcessPoolExecutor(max_workers=_ARGS.workers) as ex:
        for hx, out in ex.map(_solve_one, tasks):
            if out is None:
                continue
            results2[hx] = out
            s = out["summary"]
            d = s["opened_n"] - results[hx]["summary"]["opened_n"]
            print(f"  {hx}  [{s['status']}]  OWN served={s['own_served_pct']:6.2f}% "
                  f"of {s['own_demand_mbps']/1e3:7.1f} Gbps  "
                  f"sites={s['opened_n']:4d} ({d:+d})  own_beam={s['own_beam']} "
                  f"({s['own_ntn_mbps']:.0f} Mbps)  {s['wall_s']:.0f}s",
                  flush=True)

    # ---------------- final pass selection (MUST precede accounting) -------
    #final = results2 or results
    final, n1, n2 = {}, 0, 0
    for hx in results:
        s1 = results[hx]["summary"]
        s2 = (results2.get(hx) or {}).get("summary") if results2 else None
        if s2 is None or s1["own_served_pct"] > s2["own_served_pct"] + 1e-9:
            final[hx] = results[hx]; n1 += 1
        else:
            final[hx] = results2[hx]; n2 += 1
    print(f"\n  pass selection: {n1:,} hexes kept PASS 1, {n2:,} kept PASS 2",
          flush=True)

    # ------------- SPARSE-HEX ACCOUNTING (full-region totals) --------------
    # Hexes under --min-users get no TN build, but their demand is real and
    # the simulator will serve what it can via NTN. Account them so the
    # stitched totals cover EVERY user exactly once (comparable with the
    # simulator). Written ONCE -- the old file had two copies of this block,
    # the first of which referenced `final` before assignment (NameError).
    sp_dem = sp_ntn = sp_out = 0.0
    sp_beams = 0
    for hx in counts:
        if hx in final:
            continue
        d = float(_DEM[_HEX_IDX[hx]].sum())
        if d <= 0:
            continue
        sp_dem += d
        sp_ntn += min(d, cap_full)
        sp_out += max(d - cap_full, 0.0)
        sp_beams += 1
    if sp_beams:
        print(f"\n  sparse hexes (<{_ARGS.min_users} users, NTN-only): "
              f"{sp_beams:,} hexes, demand {sp_dem/1e3:,.1f} Gbps -> "
              f"NTN {sp_ntn/1e3:,.1f}, outage {sp_out/1e3:,.1f} Gbps",
              flush=True)

    # ---------------- STITCH ----------------
    n_sites = 0
    with open(_ARGS.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hex", "lat", "lon", "tier"])
        for hx, out in final.items():
            for site in out["opened"]:
                w.writerow([hx, f"{site['lat']:.6f}", f"{site['lon']:.6f}",
                            site["tier_name"]])
                n_sites += 1
    tot_dem = sum(o["summary"]["own_demand_mbps"] for o in final.values()) + sp_dem
    tot_tn = sum(o["summary"]["own_tn_mbps"] for o in final.values())
    tot_ntn = sum(o["summary"]["own_ntn_mbps"] for o in final.values()) + sp_ntn
    tot_out = sum(o["summary"]["own_out_mbps"] for o in final.values()) + sp_out
    beams = sum(o["summary"]["own_beam"] for o in final.values()) + sp_beams
    print(f"\n=== STITCHED (owned demand only, no halo double-counting) ==="
          f"\n  hexes={len(final)}  sites={n_sites:,}  "
          f"beams={beams} (<= 1 per hex)"
          f"\n  demand={tot_dem/1e3:,.1f} Gbps"
          f"\n  TN={tot_tn/1e3:,.1f}  NTN={tot_ntn/1e3:,.1f}  "
          f"outage={tot_out/1e3:,.1f} Gbps"
          f"  ({100*(tot_tn+tot_ntn)/max(tot_dem,1e-9):.2f}% served)"
          f"\n  -> {_ARGS.out}", flush=True)


if __name__ == "__main__":
    main()