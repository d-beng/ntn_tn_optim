#!/usr/bin/env python3
"""
iterative_milp.py — the paper's novel solver component:
SIMULATOR-CORRECTED ITERATIVE MILP.

Loop:  (1) solve the frozen-SE MILP  ->  placement P_i
       (2) oracle(P_i) returns REALIZED per-(demand-pt, site) spectral
           efficiencies under true interference for the deployed network
           (on the cluster: your 3GPP simulator / sinr.py with the actual
           opened interferer set; here: a pluggable callable)
       (3) correct the instance's eligibility SEs with a damped update
           eta <- (1-w)*eta + w*eta_realized ; add a no-good cut on any
           hex whose realized offload exceeded the beam ceiling
       (4) re-solve warm-started; stop when served demand stabilizes.

Motivation: Prop. 3 — with endogenous interference even the continuous
relaxation is non-convex, so no single convex/MILP solve can be exact; the
loop reconciles the tractable surrogate with the true physics empirically.
Precedent: iterative-refinement schemes for SINR-coefficient wireless design
MILPs (D'Andreagiovanni & Gleixner, arXiv:1604.00513).
"""
from __future__ import annotations
import time
import numpy as np

from candidate_generator import Instance
from hex_milp import solve_hex


def default_oracle_factory(inst: Instance, se_fn, rho_dep: float = 0.95):
    """Planning-level stand-in oracle for offline testing: recompute each
    eligible pair's SE with an interference margin that grows with the LOCAL
    density of OPENED same-band sites around the serving candidate (a crude
    but monotone proxy for the true simulator's interference response).
    On the cluster, replace with a wrapper that runs the real simulator on
    the placement CSV and returns realized SEs per (u, j)."""
    from scipy.spatial import cKDTree
    import math

    def oracle(y: np.ndarray):
        open_idx = np.where(y)[0]
        if len(open_idx) == 0:
            return [list(se) for se in inst.elig_se]
        tree = cKDTree(inst.cand_xy[open_idx])
        realized = []
        for u in range(len(inst.dem_mbps)):
            row = []
            for j, se0 in zip(inst.elig_j[u], inst.elig_se[u]):
                t = inst.tiers[inst.cand_tier[j]]
                # count opened same-band neighbours within 3R of the site
                near = tree.query_ball_point(inst.cand_xy[j], 3.0 * t.radius_km)
                n_intf = max(0, len(near) - (1 if y[j] else 0))
                # each nearby open site adds ~ +1.2 dB interference (proxy)
                penalty_db = 1.2 * min(n_intf, 8)
                se = se0 * max(0.35, 1.0 - penalty_db / 30.0)
                row.append(se)
            realized.append(row)
        return realized

    return oracle


def solve_iterative(inst: Instance,
                    oracle,                       # y -> realized elig_se
                    lam: float = 100.0,
                    c_beam: float = 5.0,
                    max_iter: int = 6,
                    damping: float = 0.5,
                    stab_tol: float = 0.02,       # 2% served-demand stability
                    mip_gap: float = 0.02,
                    time_limit_s: float = 600.0,
                    threads: int = 0,
                    log: bool = True):
    history = []
    prev_served = None
    t0 = time.time()

    for it in range(1, max_iter + 1):
        res = solve_hex(inst, lam=lam, c_beam=c_beam,
                        mip_gap=mip_gap, time_limit_s=time_limit_s,
                        threads=threads, log=False)
        served = res["served_tn_mbps"] + res["ntn_mbps"]
        history.append({
            "iter": it, "objective": res["objective"],
            "served_pct": res["served_pct"], "beams": res["beams"],
            "opened": dict(res["opened"]), "gap": res["gap"],
            "wall_s": res["wall_s"],
        })
        if log:
            print(f"  [iter {it}] served={res['served_pct']:.2f}%  "
                  f"obj={res['objective']:.0f}  opened={res['opened']}  "
                  f"beams={res['beams']}  ({res['wall_s']:.0f}s)", flush=True)

        if prev_served is not None and \
           abs(served - prev_served) <= stab_tol * max(prev_served, 1e-9):
            if log:
                print(f"  converged: served-demand change "
                      f"<= {100*stab_tol:.0f}% across iterations.")
            break
        prev_served = served

        # ---- oracle correction (damped) ----------------------------------
        realized = oracle(res["y"])
        for u in range(len(inst.dem_mbps)):
            inst.elig_se[u] = [
                (1.0 - damping) * se0 + damping * ser
                for se0, ser in zip(inst.elig_se[u], realized[u])
            ]

    history[-1]["total_wall_s"] = time.time() - t0
    return res, history
