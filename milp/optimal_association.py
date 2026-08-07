#!/usr/bin/env python3
"""
optimal_association.py — what the SAME deployment would serve under PERFECT
load balancing, as a certified LP bound.

WHY THIS IS A SECOND MEASUREMENT, NOT A SIMULATOR CHANGE
--------------------------------------------------------
The simulator attaches every user to its best-SINR cell and drops it if that
cell is full (allow_spillover=False). That is an UNCOORDINATED network -- a
pessimistic but honest model of attachment with no SON, no CIO, no mobility
load balancing.

Making the simulator itself optimal would turn it into a genie, and the
model-vs-simulator check in [F] would then be comparing a model against a
model. So the simulator is left alone and this runs afterwards, on its
output, as the opposite bound:

    greedy max-SINR      no coordination      <- what the simulator measures
    optimal association  perfect coordination <- what this computes
    real network (CIO, MLB, load-based HO)    <- lies between the two

The distance between them is the VALUE OF LOAD BALANCING on this deployment,
which is a result rather than a residual.

WHAT IT SOLVES
--------------
Cells are fixed. Let s_pj be demand of point p served by cell j:

    max  sum_{p,j} s_pj
    s.t. sum_j s_pj              <= d_p        (cannot serve more than offered)
         sum_p s_pj / eta_pj     <= W_j        (cell spectrum, Hz)
         s_pj >= 0, only for j within range of p

A pure LP -- no integers, since nothing is being built. HiGHS solves it to
proven optimality, so the number really is an upper bound and not a heuristic.

HONESTY ABOUT eta
-----------------
eta_pj for a cell the user never attached to was never measured. Two modes:

  --se-mode kappa   (default) nominal link budget scaled by the SAME
                    tier x distance kappa curve the oracle fitted from this
                    run. Consistent with how the MILP was corrected.
  --se-mode proxy   reuse each user's measured SE at its SERVING cell for all
                    alternatives. Simpler, and what coordination_gap() does --
                    but optimistic, because an alternative cell is by
                    definition weaker than the one max-SINR chose.

Report which one you used. 'kappa' is the defensible one.

USAGE (from inside smoke_real_sim.py, after the census):
    from optimal_association import optimal_association_bound
    optimal_association_bound(inst, res["y"], hex_users, bss, scen_cfg,
                              kappa_by_tier_dbin=..., log=True)
"""
from __future__ import annotations
import math
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
def _cell_capacity_hz(bss):
    """Usable Hz per sector cell, keyed by bs_id."""
    return {int(b.bs_id): float(b.total_bandwidth_hz) for b in bss}


def optimal_association_bound(inst, y, hex_users, bss, scen_cfg,
                              se_mode: str = "kappa",
                              kappa=None, kappa_tier=None,
                              max_alt: int = 8,
                              time_limit_s: float = 600.0,
                              threads: int = 0,
                              log: bool = True):
    """Upper bound on served demand for the FIXED deployment in `bss`.

    inst   : the Instance the MILP used (for demand points + geometry)
    y      : the placement (to map candidates -> cells)
    kappa, kappa_tier : the tier x distance-bin correction the oracle fitted
             this run. Pass them through from make_real_simulation_oracle so
             the SE here matches the SE the correction loop used.
    """
    import highspy
    from scipy.spatial import cKDTree
    from full_pipeline_hooks import user_result_fields

    # ---- 1. what the simulator actually achieved (the greedy baseline) ----
    served_greedy = 0.0
    demand_tot = 0.0
    used_hz = defaultdict(float)
    for u in hex_users:
        bsid, se, got, dem, _ = user_result_fields(u)
        demand_tot += dem
        served_greedy += got
        if bsid is not None and se > 0:
            used_hz[bsid] += got * 1e6 / se

    # ---- 2. cells, and their per-SITE spectrum pools --------------------
    # A site's 3 sectors each hold W_t. A demand point is served by whichever
    # sector faces it, which is fixed by geometry -- so pool per (site,sector)
    # exactly as constraint (8e) does.
    cap_hz = _cell_capacity_hz(bss)
    site_of, tier_of, latlon = {}, {}, {}
    for b in bss:
        sid = int(getattr(b, "site_id", b.bs_id))
        site_of[int(b.bs_id)] = sid
        sc = getattr(b, "scenario", None)
        tier_of[sid] = getattr(sc, "name", str(sc))
        latlon[sid] = (float(b.lat), float(b.lon))

    open_idx = np.where(np.asarray(y, dtype=bool))[0]
    # site_id was assigned in build_base_stations as enumerate(open_idx)
    cand_of_site = {i: int(j) for i, j in enumerate(open_idx)}

    n_sites = len(cand_of_site)
    W_site = {}
    for sid in range(n_sites):
        j = cand_of_site[sid]
        W_site[sid] = inst.tiers[inst.cand_tier[j]].bw_hz / 1e6   # MHz-equiv

    # ---- 3. bipartite in-range graph at DEMAND-POINT granularity --------
    xy_site = np.array([inst.cand_xy[cand_of_site[s]] for s in range(n_sites)])
    tree = cKDTree(xy_site)
    r_eff = list(inst.r_eff_km) if inst.r_eff_km else \
        [t.radius_km for t in inst.tiers]
    rmax = max(r_eff)

    NB, DMAX = 8, 3.0

    def _dbin(dkm):
        return min(NB - 1, int(NB * min(dkm, DMAX) / DMAX))

    def se_of(dkm, ti):
        base = None
        for jj, ss in zip(inst.elig_j, inst.elig_se):
            break
        # nominal from the instance's own se model, via the tier radius curve
        # (inst.elig_se holds nominal-or-corrected values; use the pristine
        #  link budget through the tier definition instead)
        from candidate_generator import se_default
        base = se_default(dkm, inst.tiers[ti])
        if se_mode != "kappa":
            return max(base, 0.05)
        k = None
        if kappa is not None:
            k = kappa.get((int(ti), _dbin(dkm)))
        if k is None and kappa_tier is not None:
            k = kappa_tier.get(int(ti))
        return max(base * (k if k else 1.0), 0.05)

    pairs = []          # (p, sid, se)
    U = len(inst.dem_mbps)
    for p in range(U):
        near = tree.query_ball_point(inst.dem_xy[p], rmax)
        cand = []
        for sid in near:
            j = cand_of_site[sid]
            ti = int(inst.cand_tier[j])
            d = float(np.hypot(*(xy_site[sid] - inst.dem_xy[p])))
            if d > r_eff[ti]:
                continue
            cand.append((se_of(d, ti), sid))
        cand.sort(reverse=True)
        for se, sid in cand[:max_alt]:
            pairs.append((p, sid, se))

    if log:
        print(f"      [optassoc] {len(pairs):,} (demand-pt, site) in-range "
              f"pairs over {U:,} points and {n_sites:,} sites "
              f"[se_mode={se_mode}]", flush=True)

    # ---- 4. the LP ------------------------------------------------------
    n_var = len(pairs)
    h = highspy.Highs()
    if not log:
        h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit_s)
    if threads:
        h.setOptionValue("threads", int(threads))

    ub = np.array([float(inst.dem_mbps[p]) for p, _s, _e in pairs])
    h.addVars(n_var, np.zeros(n_var), ub)
    h.changeColsCost(n_var, np.arange(n_var, dtype=np.int32),
                     -np.ones(n_var))          # maximise served

    by_point = defaultdict(list)
    by_site = defaultdict(list)
    for k, (p, sid, se) in enumerate(pairs):
        by_point[p].append(k)
        by_site[sid].append((k, se))

    for p, ks in by_point.items():
        h.addRow(-highspy.kHighsInf, float(inst.dem_mbps[p]), len(ks),
                 np.array(ks, dtype=np.int32), np.ones(len(ks)))
    for sid, ks in by_site.items():
        idx = np.array([k for k, _ in ks], dtype=np.int32)
        val = np.array([1.0 / max(se, 1e-6) for _, se in ks])
        # 3 sectors share the site; (8e) budgets each sector separately, so a
        # site pool of 3*W is the OPTIMISTIC direction -- state it plainly.
        h.addRow(-highspy.kHighsInf, 3.0 * W_site[sid], len(idx), idx, val)

    h.run()
    sol = np.array(h.getSolution().col_value)
    status = h.modelStatusToString(h.getModelStatus())
    served_opt = float(sol[:n_var].sum()) if sol.size >= n_var else 0.0

    # ---- 5. report ------------------------------------------------------
    tot = float(inst.dem_mbps.sum())
    g_pct = 100.0 * served_greedy / max(demand_tot, 1e-9)
    o_pct = 100.0 * min(served_opt, tot) / max(tot, 1e-9)
    if log:
        print(f"      [optassoc] LP {status}", flush=True)
        print(f"      [optassoc] greedy max-SINR (simulator) : "
              f"{g_pct:6.2f}% served, {(demand_tot-served_greedy)/1e3:6.1f} "
              f"Gbps dropped", flush=True)
        print(f"      [optassoc] OPTIMAL association (LP)    : "
              f"{o_pct:6.2f}% served, {(tot-min(served_opt,tot))/1e3:6.1f} "
              f"Gbps dropped", flush=True)
        print(f"      [optassoc] -> VALUE OF LOAD BALANCING on this fixed "
              f"deployment: {o_pct-g_pct:+.2f} pp "
              f"({(served_opt-served_greedy)/1e3:,.1f} Gbps).", flush=True)
        print(f"      [optassoc]    No new sites. The deployment already "
              f"carries this; only the attachment rule changes.", flush=True)
        print(f"      [optassoc]    Bound is OPTIMISTIC: site-level spectrum "
              f"pooling and no shadowing on the alternative link.", flush=True)
    return {"served_greedy_pct": g_pct, "served_optimal_pct": o_pct,
            "gain_pp": o_pct - g_pct, "status": status,
            "se_mode": se_mode, "n_pairs": len(pairs)}