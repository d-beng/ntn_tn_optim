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
    export PYTHONPATH=/Utilisateurs/dbenguer/ntn_tn_optim/src
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
_HEX_IDX = None            # hex_id -> np.array of user indices (built ONCE)
_TIERS = _SE_FN = _SCEN = None
_ARGS = None
_ORACLE_KIND = None      # "real" | "proxy" | None (decided once per worker)


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
                          agg_res=_ARGS.agg_res, K_elig=_ARGS.k_elig,
                          beam_bw_hz=_ARGS.beam_bw, beam_se=_ARGS.beam_se)
    import inspect as _insp
    _sig = _insp.signature(solve_hex).parameters
    use_min_out = getattr(_ARGS, "min_outage", False)
    if use_min_out:
        from hex_milp import solve_hex_min_outage
        solver_fn, skw = solve_hex_min_outage, {}
    else:
        solver_fn = solve_hex
        skw = {}
        if getattr(_ARGS, "max_outage_mbps", None) is not None \
                and "outage_budget_mbps" in _sig:
            skw["outage_budget_mbps"] = float(_ARGS.max_outage_mbps)
        if "max_outage_frac" in _sig:          # newer hex_milp.py only
            skw["max_outage_frac"] = (0.0 if _ARGS.zero_outage else None)
        elif _ARGS.zero_outage:
            raise SystemExit("--zero-outage needs the current hex_milp.py "
                             "(deploy the latest file)")

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

    # ---- OWNED-DEMAND accounting: the ONLY figures that may be summed -----
    # Each hex model also contains HALO users from ring-1 neighbours; those
    # users are solved again (and served) in their own hex's model. Summing a
    # model's totals counts border users up to 7x, and reports halo users as
    # "unserved" here whenever the neighbour's beam residual is exhausted --
    # even though their owner hex serves them.
    own_u = [u for u in range(len(inst.dem_mbps)) if inst.dem_hex[u] == hex_id]
    own_set = set(own_u)
    own_d = float(sum(inst.dem_mbps[u] for u in own_u))
    own_tn = 0.0
    for (u, j), col in res["x_index"].items():
        if u in own_set:
            own_tn += float(res["x_val"][col]) * float(inst.dem_mbps[u])
    own_out = float(sum(res["o"][u] * inst.dem_mbps[u] for u in own_u))
    summary.update({"own_demand_mbps": own_d, "own_tn_mbps": own_tn,
                    "own_out_mbps": own_out, "n_own_dem_pts": len(own_u)})

    # ---- OWNED-DEMAND accounting (the only figures that may be summed) ----
    # Each hex model also contains HALO users from ring-1 neighbours; those
    # users are solved again (and served) in their own hex's model. Summing
    # a model's totals would therefore count border users up to 7 times, and
    # would report halo users as "unserved" here whenever the neighbour's beam
    # residual is exhausted -- even though their owner hex serves them.
    # We therefore report, and stitch, ONLY demand points owned by this hex.
    own_u = [u for u in range(len(inst.dem_mbps)) if inst.dem_hex[u] == hex_id]
    own_d = float(sum(inst.dem_mbps[u] for u in own_u))
    own_set = set(own_u)
    own_tn = 0.0
    for (u, j), col in res["x_index"].items():
        if u in own_set:
            own_tn += float(res["x_val"][col]) * float(inst.dem_mbps[u])
    own_ntn = float(sum(res["z"][u] * inst.dem_mbps[u] for u in own_u))
    own_out = float(sum(res["o"][u] * inst.dem_mbps[u] for u in own_u))
    summary.update({
        "own_demand_mbps": own_d,
        "own_tn_mbps": own_tn,
        "own_out_mbps": own_out,
        "own_served_pct": 100.0 * (own_tn + own_ntn) / max(own_d, 1e-9),
        "n_own_dem_pts": len(own_u),
    })
    # OWNED beam accounting: this solve owns ONLY its centre hex's beam.
    # Halo users' offload onto neighbours' beams is reported separately and
    # never summed at stitch time (that beam belongs to the neighbour's solve).
    summary["own_ntn_mbps"] = own_ntn
    summary["own_beam"] = 1 if own_ntn > 1e-9 else 0
    summary["halo_ntn_mbps"] = float(
        sum(v for k, v in res["ntn_by_hex"].items() if k != hex_id))
    from hex_milp import diagnose_outage
    summary["why"] = diagnose_outage(inst, res)
    if "history" in res:
        h0, hN = res["history"][0], res["history"][-1]
        summary["iters"] = len(res["history"])
        summary["se_drift"] = hN["mean_se"] - h0["mean_se"]
        summary["served_drift"] = hN["served_pct"] - h0["served_pct"]
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
    ap.add_argument("--config", default="/Utilisateurs/dbenguer/ntn_tn_optim/configs/5g_base.yaml")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-users", type=int, default=500)
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
    ap.add_argument("--lam", type=float, default=1309.0)
    ap.add_argument("--c-beam", type=float, default=49813.0)
    ap.add_argument("--gap", type=float, default=0.02)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--beam-bw", type=float, default=300e6)
    ap.add_argument("--beam-se", type=float, default=1.77)
    ap.add_argument("--max-outage-mbps", type=float, default=None,
                    help="HARD cap on total outage per hex [Mbps] "
                         "(eps-constraint, Eq. 8b). e.g. 10 for near-zero. "
                         "Infeasible => that hex cannot meet the target.")
    ap.add_argument("--max-outage-mbps", type=float, default=None,
                    help="HARD cap on total outage per hex [Mbps] "
                         "(eps-constraint, Eq. 8b). e.g. 10 for near-zero; "
                         "Infeasible => that hex cannot meet the target.")
    ap.add_argument("--iterate", type=int, default=1,
                    help="physics-corrected iterations per hex (1 = single "
                         "shot; 3-5 recommended for the full loop)")
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--stab-tol", type=float, default=0.02)
    ap.add_argument("--min-outage", action="store_true",
                    help="lexicographic: minimise outage first, then cost "
                         "(always returns a placement + the outage floor)")
    ap.add_argument("--zero-outage", action="store_true",
                    help="forbid outage entirely; INFEASIBLE proves saturation")
    ap.add_argument("--out", default="province_placement.csv")
    _ARGS = ap.parse_args()

    # tiers + real SE. load_tiers/make_real_se_fn now take the SCENARIOS
    # config object (same as smoke_real_sim.py), not a path.
    from omegaconf import OmegaConf
    from run_real_tile import load_tiers, make_real_se_fn
    _cfg = OmegaConf.load(_ARGS.config)
    _scen = _cfg.terrestrial.scenarios if "terrestrial" in _cfg else _cfg.scenarios
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

    # ---------------- SPARSE-HEX ACCOUNTING (full-Ontario totals) ----------
    # Hexes under --min-users get no TN build; their users go to their hex
    # beam: served = min(demand, C_beam), remainder = outage. Included so the
    # stitched totals cover ALL of Ontario, comparable with the simulator.
    cap_full = _ARGS.beam_bw * _ARGS.beam_se / 1e6
    sparse = {hx: n for hx, n in counts.items()
              if hx not in final and n > 0}
    sp_dem = sp_ntn = sp_out = 0.0
    sp_beams = 0
    for hx in sparse:
        idx = _HEX_IDX[hx]
        d = float(_DEM[idx].sum())
        if d <= 0:
            continue
        served = min(d, cap_full)
        sp_dem += d
        sp_ntn += served
        sp_out += d - served
        sp_beams += 1
    if sparse:
        print(f"\n  sparse hexes (<{_ARGS.min_users} users, NTN-only): "
              f"{len(sparse):,} hexes, demand {sp_dem/1e3:,.1f} Gbps -> "
              f"NTN {sp_ntn/1e3:,.1f}, outage {sp_out/1e3:,.1f} Gbps, "
              f"{sp_beams:,} beams", flush=True)

    # ------------- SPARSE-HEX ACCOUNTING (full-region totals) ------------
    # Hexes under --min-users get no TN build, but their demand is real and
    # the simulator will serve it via NTN. Account them so the stitched
    # totals cover EVERY user exactly once (comparable with the simulator).
    cap_full = _ARGS.beam_bw * _ARGS.beam_se / 1e6
    sp_dem = sp_ntn = sp_out = 0.0
    sp_beams = 0
    for hx in counts:
        if hx in (results2 or results):
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