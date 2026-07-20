#!/usr/bin/env python3
"""
province_solver.py — multi-hex decomposition driver.

PASS 1  every dense hex solved independently (parallel across cores):
        hex + halo users, own candidates only -> placement + per-sector usage.
PASS 2  Jacobi sweep: each hex re-solved with neighbors' PASS-1 border towers
        visible as FIXED-OPEN, ZERO-COST candidates carrying their RESIDUAL
        per-sector capacity -> border overbuild pruned, edge users spill onto
        neighbors' spare spectrum. (One sweep suffices: coupling band is
        ~R_max wide vs a 16 km hex.)
STITCH  owned opened sites from the final pass -> province placement CSV
        (feed to the simulator for full-physics validation).

Usage (cluster):
    export PYTHONPATH=/home/db3n/Documents/Ph.D./Courses/ntn_tn_test/ntn_tn_optim-master_2/src
    python province_solver.py --users .../users.pkl --config .../5g_base.yaml \\
        --workers 32 --min-users 500 [--max-hexes 20]   # start small!
On the first run use --max-hexes 5 to validate the loop before the full set.
"""
from __future__ import annotations
import argparse, csv, math, os, pickle, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import h3

from candidate_generator import build_instance, unproject
from hex_milp import solve_hex, sector_usage_mhz

# globals for fork-shared arrays (same CoW pattern as the simulator fix)
_LAT = _LON = _DEM = None
_CELLS = None
_TIERS = _SE_FN = None
_ARGS = None


def _hav_km(a1, b1, a2, b2):
    p1, p2 = math.radians(a1), math.radians(a2)
    dp, dl = p2 - p1, math.radians(b2 - b1)
    x = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * 6371.0088 * math.asin(math.sqrt(x))


def _extract(hex_id):
    """hex + halo user arrays from the fork-shared globals."""
    clat, clon = h3.cell_to_latlng(hex_id)
    edge = h3.average_hexagon_edge_length(5, unit="km")
    r_cut = edge + _ARGS.halo_km
    neigh = set(h3.grid_disk(hex_id, 1))
    idx = [i for i, c in enumerate(_CELLS)
           if c == hex_id or (c in neigh and
                              _hav_km(_LAT[i], _LON[i], clat, clon) <= r_cut)]
    idx = np.array(idx, dtype=np.int64)
    return _LAT[idx], _LON[idx], _DEM[idx]


def _solve_one(task):
    """PASS-1 or PASS-2 solve for one hex. task = (hex_id, ext_sites|None)."""
    hex_id, ext = task
    lat, lon, dem = _extract(hex_id)
    m = dem >= 0.1
    if m.sum() < _ARGS.min_users:
        return hex_id, None
    inst = build_instance(lat[m], lon[m], dem[m], hex_id,
                          tiers=_TIERS, se_fn=_SE_FN,
                          ext_sites=ext,
                          rho_cand=_ARGS.rho_cand, rho_dep=_ARGS.rho_dep,
                          agg_res=_ARGS.agg_res, K_elig=_ARGS.k_elig,
                          beam_bw_hz=_ARGS.beam_bw, beam_se=_ARGS.beam_se)
    res = solve_hex(inst, lam=_ARGS.lam, c_beam=_ARGS.c_beam,
                    mip_gap=_ARGS.gap, time_limit_s=_ARGS.time_limit,
                    threads=1, log=False)
    usage = sector_usage_mhz(inst, res)
    y = res["y"]
    opened = []
    for j in np.where(y)[0]:
        if inst.fixed_open[j]:
            continue                        # neighbor's site, not ours
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
    return hex_id, {"opened": opened, "summary": summary}


def main():
    global _LAT, _LON, _DEM, _CELLS, _TIERS, _SE_FN, _ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="/home/db3n/Documents/Ph.D./Courses/ntn_tn_test/ntn_tn_optim-master_2/data/users.pkl")
    ap.add_argument("--config", default="/home/db3n/Documents/Ph.D./Courses/ntn_tn_test/ntn_tn_optim-master_2/configs/terrestrial/5g_base.yaml")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-users", type=int, default=0)
    ap.add_argument("--max-hexes", type=int, default=0, help="0 = all")
    ap.add_argument("--hour", type=float, default=20.0)
    ap.add_argument("--halo-km", type=float, default=2.847)
    ap.add_argument("--rho-cand", type=float, default=1.0)
    ap.add_argument("--rho-dep", type=float, default=0.95)
    ap.add_argument("--agg-res", type=int, default=9)
    ap.add_argument("--k-elig", type=int, default=6)
    ap.add_argument("--dens-uma", type=float, default=1600.0)
    ap.add_argument("--dens-umi", type=float, default=4000.0)
    ap.add_argument("--dens-mmw", type=float, default=15000.0)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--c-beam", type=float, default=5.0)
    ap.add_argument("--gap", type=float, default=0.02)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--beam-bw", type=float, default=300e6)
    ap.add_argument("--beam-se", type=float, default=1.77)
    ap.add_argument("--out", default="province_placement.csv")
    _ARGS = ap.parse_args()

    # tiers + real SE from the existing cluster bridge
    from run_real_tile import load_tiers, make_real_se_fn
    _TIERS = load_tiers(_ARGS.config, _ARGS.dens_uma, _ARGS.dens_umi,
                        _ARGS.dens_mmw)
    _SE_FN = make_real_se_fn(_ARGS.config, rho_dep=_ARGS.rho_dep)

    print(f"loading {_ARGS.users} ...", flush=True)
    t0 = time.time()
    with open(_ARGS.users, "rb") as f:
        users = pickle.load(f)
    _LAT = np.array([u.home_lat for u in users])
    _LON = np.array([u.home_lon for u in users])
    _DEM = np.array([u.get_demand_at_time(_ARGS.hour) for u in users])
    del users
    print(f"  {len(_LAT):,} users in {time.time()-t0:.0f}s", flush=True)

    _CELLS = [h3.latlng_to_cell(float(a), float(b), 5)
              for a, b in zip(_LAT, _LON)]
    counts = Counter(_CELLS)
    hexes = [hx for hx, n in counts.most_common() if n >= _ARGS.min_users]
    if _ARGS.max_hexes:
        hexes = hexes[:_ARGS.max_hexes]
    print(f"  {len(hexes)} hexes with >= {_ARGS.min_users} users", flush=True)

    # warm the SE cache once in the parent (fork-shared afterwards)
    for t in _TIERS:
        for dkm in np.linspace(0.02, t.radius_km, 30):
            _SE_FN(float(dkm), t)

    # ---------------- PASS 1 ----------------
    print("\n=== PASS 1 (independent hexes) ===", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=_ARGS.workers) as ex:
        for hx, out in ex.map(_solve_one, [(hx, None) for hx in hexes]):
            if out is None:
                continue
            results[hx] = out
            s = out["summary"]
            print(f"  {hx}  served={s['served_pct']:6.2f}%  "
                  f"sites={s['opened_n']:4d}  beams={s['beams']}  "
                  f"gap={s['gap']:.3f}  {s['wall_s']:.0f}s", flush=True)

    # ---------------- PASS 2 (Jacobi border sweep) ----------------
    print("\n=== PASS 2 (residual-capacity border sweep) ===", flush=True)
    edge = h3.average_hexagon_edge_length(5, unit="km")

    def border_ext(hex_id):
        """neighbors' PASS-1 opened sites within halo of this hex."""
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

    tasks = [(hx, border_ext(hx)) for hx in results]
    results2 = {}
    with ProcessPoolExecutor(max_workers=_ARGS.workers) as ex:
        for hx, out in ex.map(_solve_one, tasks):
            if out is None:
                continue
            results2[hx] = out
            s = out["summary"]
            d = s["opened_n"] - results[hx]["summary"]["opened_n"]
            print(f"  {hx}  served={s['served_pct']:6.2f}%  "
                  f"sites={s['opened_n']:4d} ({d:+d})  beams={s['beams']}  "
                  f"{s['wall_s']:.0f}s", flush=True)

    # ---------------- STITCH ----------------
    final = results2 or results
    n_sites = 0
    with open(_ARGS.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hex", "lat", "lon", "tier"])
        for hx, out in final.items():
            for site in out["opened"]:
                w.writerow([hx, f"{site['lat']:.6f}", f"{site['lon']:.6f}",
                            site["tier_name"]])
                n_sites += 1
    tot_tn = sum(o["summary"]["served_tn_mbps"] for o in final.values())
    tot_ntn = sum(o["summary"]["ntn_mbps"] for o in final.values())
    tot_out = sum(o["summary"]["outage_mbps"] for o in final.values())
    beams = sum(o["summary"]["beams"] for o in final.values())
    print(f"\n=== STITCHED ===\n  hexes={len(final)}  sites={n_sites:,}  "
          f"beams={beams}\n  TN={tot_tn/1e3:,.1f} Gbps  NTN={tot_ntn/1e3:,.1f} "
          f"Gbps  outage={tot_out/1e3:,.1f} Gbps\n  -> {_ARGS.out}", flush=True)


if __name__ == "__main__":
    main()
