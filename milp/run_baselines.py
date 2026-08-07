#!/usr/bin/env python3
"""
run_baselines.py — the paper's benchmark ladder, on ONE instance.

Every method sees the IDENTICAL Instance: same candidates, same eligibility,
same SE coefficients, same conflict rules, same NTN ceiling, same cost model.
Anything else is not a comparison.

    MILP     exact, with a PROVEN optimality gap        <- the reference
    greedy   lazy greedy net-benefit (pipeline analogue)
    GA       tournament GA with conflict repair          (literature baseline)
    kmeans   k centroids, k taken from the MILP's site count
             -- the naive geometric baseline: it needs k as an INPUT, has no
                capacity model, no tier structure and no NTN ceiling, so it
                cannot express the problem. That is the point of including it.

The MILP's certified gap is what makes this table mean something: greedy and
GA report an objective with NO bound, so "greedy is X% worse" is only a
statement because the MILP proved how far from optimal it could possibly be.

USAGE (same flags as the smoke test, so the instance matches exactly):
    export PYTHONPATH=/Utilisateurs/dbenguer/ntn_tn_optim/src
    python run_baselines.py workers=200 \\
        +dens_uma=999999 +dens_umi=0 +k_elig=12 \\
        +agg_res=10 +agg_safety=1.0 +rho_dep=0.75 +rho_cand=0.85 \\
        +cross_tier_m=legacy +methods=milp,greedy,kmeans

NOTE ON COST
    greedy is minutes. kmeans is seconds. The GA evaluates a full association
    pass per fitness call: at 73k demand points that is ~1-2 s each, so
    pop x gens = 20 x 25 is already ~15 min and 30 x 40 is over an hour. It is
    OFF by default; add it explicitly with +methods=...,ga and use
    +ga_pop / +ga_gens to control the budget.

    No simulation is run here. This table is MODEL-side only: it compares how
    well each method solves the SAME optimisation problem. Simulated drops for
    the winning placement come from smoke_real_sim.py.
"""
from __future__ import annotations
import json
import math
import pickle
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, ".")

DEF_USERS = "/Utilisateurs/dbenguer/ntn_tn_optim/data/users.pkl"


def _hav_km(a1, b1, a2, b2):
    p1, p2 = math.radians(a1), math.radians(a2)
    dp, dl = p2 - p1, math.radians(b2 - b1)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(x))


# ---------------------------------------------------------------------------
def solve_kmeans(inst, k, lam, c_beam, seed=0, log=True):
    """NAIVE GEOMETRIC BASELINE.

    k-means on demand-point positions weighted by demand, then snap each
    centroid to its nearest CANDIDATE (so the placement is legal on the same
    candidate set), drop centroids that violate a conflict, and evaluate with
    the same association evaluator every other method uses.

    This is the honest way to include it: k-means cannot express capacity,
    tiers, or the NTN ceiling, and it requires k as an input rather than
    deciding it. Its role is to show what a geometry-only method gives up.
    """
    from scipy.cluster.vq import kmeans2
    from scipy.spatial import cKDTree
    from baselines import _pack_result

    t0 = time.time()
    rng = np.random.default_rng(seed)
    w = inst.dem_mbps / inst.dem_mbps.sum()
    idx = rng.choice(len(inst.dem_xy), size=min(60000, len(inst.dem_xy)),
                     replace=True, p=w)                # demand-weighted sample
    pts = inst.dem_xy[idx]
    cent, _ = kmeans2(pts, k, minit="++", seed=seed, iter=25)

    tree = cKDTree(inst.cand_xy)
    _, near = tree.query(cent)
    y = np.zeros(len(inst.cand_tier), dtype=bool)
    for j in np.unique(near):
        y[int(j)] = True

    # repair conflicts (keep the first of each conflicting pair)
    for a, b in inst.conflict_pairs:
        if y[int(a)] and y[int(b)]:
            y[int(b)] = False
    if log:
        print(f"    [kmeans] k={k:,} requested -> {int(y.sum()):,} distinct "
              f"legal candidates ({time.time()-t0:.0f}s)", flush=True)
    return _pack_result(inst, y, lam, c_beam, time.time() - t0, "kmeans")


# ---------------------------------------------------------------------------
@hydra.main(version_base=None,
            config_path="/Utilisateurs/dbenguer/ntn_tn_optim/configs",
            config_name="base")
def main(cfg: DictConfig):
    import h3
    from candidate_generator import build_instance
    from hex_milp import solve_hex, LAM_DEFAULT, C_BEAM_DEFAULT
    from run_real_tile import load_tiers, make_real_se_fn

    lam = float(cfg.get("lam", LAM_DEFAULT))
    c_beam = float(cfg.get("c_beam", C_BEAM_DEFAULT))
    rho_dep = float(cfg.get("rho_dep", 0.95))
    rho_cand = float(cfg.get("rho_cand", 1.0))
    agg_res = int(cfg.get("agg_res", 10))
    agg_safety = float(cfg.get("agg_safety", 1.0))
    k_elig = int(cfg.get("k_elig", 12))
    threads = int(cfg.get("solver_threads", 64))
    tl = float(cfg.get("solve_time_limit", 21600))
    methods = [m.strip().lower() for m in
               str(cfg.get("methods", "milp,greedy,kmeans")).split(",")]
    xt_m = cfg.get("cross_tier_m", "legacy")
    if isinstance(xt_m, str) and xt_m.lower() in ("none", "null", "off"):
        xt_m = None

    # ---------- users ----------
    users_path = str(cfg.get("users", DEF_USERS))
    print(f"[.] loading {users_path} ...", flush=True)
    t0 = time.time()
    with open(users_path, "rb") as f:
        users = pickle.load(f)
    print(f"    {len(users):,} users in {time.time()-t0:.0f}s", flush=True)
    for u in users:
        u.move(20.0, 5)
    cells = {}
    for i, u in enumerate(users):
        cells.setdefault(h3.latlng_to_cell(u.current_lat, u.current_lon, 5),
                         []).append(i)
    hex_id = cfg.get("hex") or max(cells, key=lambda c: len(cells[c]))
    neigh = set(h3.grid_disk(hex_id, 1))
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
            sel.extend(cells[c] if c == hex_id else
                       [i for i in cells[c]
                        if _hav_km(users[i].current_lat, users[i].current_lon,
                                   clat, clon) <= r_cut])
    hu = [users[i] for i in sel]
    del users
    lat = np.array([u.current_lat for u in hu])
    lon = np.array([u.current_lon for u in hu])
    dem = np.array([u.get_demand_at_time(20.0) for u in hu])
    del hu
    print(f"    hex {hex_id}: {len(lat):,} users, {dem.sum()/1e3:.1f} Gbps",
          flush=True)

    # ---------- the ONE instance every method sees ----------
    scen = cfg.terrestrial.scenarios
    tiers = load_tiers(scen, float(cfg.get("dens_uma", 1600)),
                       float(cfg.get("dens_umi", 4000)),
                       float(cfg.get("dens_mmw", 15000)))
    se_fn = make_real_se_fn(scen, rho_dep=rho_dep)
    _xt = ({"cross_tier_midband_conflict": True}
           if isinstance(xt_m, str) and xt_m.lower() == "legacy"
           else {"cross_tier_min_dist_km":
                 (None if xt_m is None else float(xt_m) / 1000.0)})
    inst = build_instance(lat, lon, dem, hex_id, tiers=tiers, se_fn=se_fn,
                          rho_cand=rho_cand, rho_dep=rho_dep, K_elig=k_elig,
                          agg_res=agg_res, agg_safety=agg_safety, **_xt)
    print(f"    instance: {len(inst.cand_tier):,} candidates, "
          f"{len(inst.dem_mbps):,} demand pts, "
          f"{sum(len(e) for e in inst.elig_j):,} x-cols, "
          f"{len(inst.conflict_pairs):,} conflicts", flush=True)
    print(f"    cost model: lambda={lam:,.0f} USD/Mbps/yr, "
          f"c_beam={c_beam:,.0f} USD/yr", flush=True)

    rows = []

    def record(name, r, note=""):
        rows.append({
            "method": name, "objective": r["objective"],
            "gap": r.get("gap", float("nan")),
            "sites": int(sum(r["opened"].values())),
            "opened": json.dumps(r["opened"]), "beams": r["beams"],
            "served_pct": r["served_pct"],
            "tn_mbps": r["served_tn_mbps"], "ntn_mbps": r["ntn_mbps"],
            "outage_mbps": r["outage_mbps"], "wall_s": r.get("wall_s", 0.0),
            "note": note,
        })
        print(f"  == {name:<8} obj={r['objective']:>14,.0f}  "
              f"sites={int(sum(r['opened'].values())):>6,}  "
              f"served={r['served_pct']:>6.2f}%  "
              f"{r.get('wall_s', 0):>7.1f}s  {note}", flush=True)

    # ---------- MILP: the reference ----------
    k_milp = None
    if "milp" in methods:
        print("\n[MILP] exact reference", flush=True)
        r = solve_hex(inst, lam=lam, c_beam=c_beam, mip_gap=0.02,
                      time_limit_s=tl, threads=threads, log=True)
        k_milp = int(sum(r["opened"].values()))
        record("milp", r, f"PROVEN gap {r['gap']:.2%} ({r['status']})")

    # ---------- greedy ----------
    if "greedy" in methods:
        print("\n[greedy] lazy net-benefit", flush=True)
        from baselines import solve_greedy
        record("greedy", solve_greedy(inst, lam=lam, c_beam=c_beam, log=True))

    # ---------- k-means ----------
    if "kmeans" in methods:
        k = int(cfg.get("kmeans_k", k_milp or 1000))
        print(f"\n[kmeans] naive geometric, k={k:,} (taken from the MILP)",
              flush=True)
        record("kmeans", solve_kmeans(inst, k, lam, c_beam),
               f"k SUPPLIED, not decided")

    # ---------- GA ----------
    if "ga" in methods:
        pop = int(cfg.get("ga_pop", 20)); gens = int(cfg.get("ga_gens", 25))
        print(f"\n[GA] pop={pop} gens={gens} "
              f"({pop*gens:,} full association passes -- slow)", flush=True)
        from baselines import solve_ga
        record("ga", solve_ga(inst, lam=lam, c_beam=c_beam, pop=pop,
                              gens=gens, log=True))

    # ---------- table ----------
    import csv
    out = f"baselines_{hex_id[:8]}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    base = next((r for r in rows if r["method"] == "milp"), rows[0])
    print("\n=== BENCHMARK LADDER ===")
    print(f"{'method':<9}{'objective':>15}{'vs MILP':>10}{'sites':>8}"
          f"{'served%':>9}{'outage Gbps':>13}{'wall_s':>9}")
    for r in rows:
        d = 100.0 * (r["objective"] - base["objective"]) / max(base["objective"], 1e-9)
        print(f"{r['method']:<9}{r['objective']:>15,.0f}{d:>+9.1f}%"
              f"{r['sites']:>8,}{r['served_pct']:>9.2f}"
              f"{r['outage_mbps']/1e3:>13.2f}{r['wall_s']:>9.1f}")
    print(f"\n  MILP gap is PROVEN: no deployment on this candidate set beats "
          f"{base['objective']:,.0f} by more than {base['gap']:.2%}.")
    print(f"  Heuristics report no bound at all -- that asymmetry is why the "
          f"exact solver is in the ladder.")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()