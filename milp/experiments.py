#!/usr/bin/env python3
"""
experiments.py — run ALL methods on one instance and produce the paper's
comparison table + figures:
    MILP (coarse)          exact reference (proven gap)
    MILP + refinement      pixel-precision variant
    Iterative (oracle)     the simulator-corrected loop   [optional]
    Greedy                 density-driven baseline (pipeline analogue)
    GA                     metaheuristic baseline

FIXES IN THIS REVISION
----------------------
* lam / c_beam defaulted to 100 / 5. Every row of the comparison table was
  therefore scored against a DIFFERENT cost model from the rest of the
  project. Now imported from hex_milp -- one source of truth.
* `from iterative_milp import ...` at module scope made the whole file
  unimportable if that module is missing or broken. Optional methods are now
  imported lazily and skipped with a printed note.
* run_all returned `hist` from a method that may not have run -> NameError.

Outputs: results.csv + fig_pareto.png + fig_convergence.png
Usage:   python experiments.py            (synthetic tile)
         import and call run_all(inst, user_xy, se_fn) with real data.
"""
from __future__ import annotations
import copy, json, time
import numpy as np

from candidate_generator import (build_instance, DEFAULT_TIERS, se_default,
                                 project_km)
from hex_milp import solve_hex, LAM_DEFAULT, C_BEAM_DEFAULT
from baselines import solve_greedy, solve_ga
from refine import solve_with_refinement


def run_all(inst, user_xy, se_fn, lam=LAM_DEFAULT, c_beam=C_BEAM_DEFAULT,
            mip_gap=0.02, time_limit_s=600.0, threads=0,
            with_iterative: bool = True):
    rows = []
    hist = []

    def add(name, res, extra=None):
        rows.append({
            "method": name,
            "objective": res["objective"],
            "served_pct": res["served_pct"],
            "tn_mbps": res["served_tn_mbps"],
            "ntn_mbps": res["ntn_mbps"],
            "outage_mbps": res["outage_mbps"],
            "beams": res["beams"],
            "sites": int(sum(res["opened"].values())),
            "opened": json.dumps(res["opened"]),
            "gap": res.get("gap", float("nan")),
            "wall_s": res.get("wall_s", float("nan")),
            **(extra or {}),
        })

    print("== MILP (coarse) ==")
    r_milp = solve_hex(copy.deepcopy(inst), lam=lam, c_beam=c_beam,
                       mip_gap=mip_gap, time_limit_s=time_limit_s,
                       threads=threads, log=False)
    add("milp_coarse", r_milp)

    print("== MILP + refinement ==")
    _, r_fine, _ = solve_with_refinement(copy.deepcopy(inst), user_xy, se_fn,
                                         lam=lam, c_beam=c_beam,
                                         mip_gap=mip_gap,
                                         time_limit_s=time_limit_s,
                                         threads=threads, log=True)
    add("milp_refined", r_fine)

    if with_iterative:
        try:
            from iterative_milp import solve_iterative, default_oracle_factory
        except Exception as e:
            print(f"== Iterative: SKIPPED ({e}) ==")
        else:
            print("== Iterative simulator-corrected MILP ==")
            inst_it = copy.deepcopy(inst)
            oracle = default_oracle_factory(inst_it, se_fn)
            r_iter, hist = solve_iterative(inst_it, oracle, lam=lam,
                                           c_beam=c_beam, mip_gap=mip_gap,
                                           time_limit_s=time_limit_s,
                                           threads=threads, log=True)
            add("iterative", r_iter, {"iters": len(hist)})

    print("== Greedy ==")
    r_g = solve_greedy(copy.deepcopy(inst), lam=lam, c_beam=c_beam)
    add("greedy", r_g)

    print("== GA ==")
    r_ga = solve_ga(copy.deepcopy(inst), lam=lam, c_beam=c_beam,
                    pop=30, gens=40)
    add("ga", r_ga)

    return rows, hist


def save_outputs(rows, hist, prefix=""):
    import csv
    with open(f"{prefix}results.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"-> {prefix}results.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for r in rows:
            ax.scatter(r["sites"], r["served_pct"], s=70)
            ax.annotate(r["method"], (r["sites"], r["served_pct"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("terrestrial sites opened")
        ax.set_ylabel("served demand [%]")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}fig_pareto.png", dpi=160)
        print(f"-> {prefix}fig_pareto.png")

        if hist:
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.plot([h["iter"] for h in hist],
                    [h.get("served_pct", np.nan) for h in hist], "o-")
            ax.set_xlabel("iteration")
            ax.set_ylabel("served demand [%]")
            ax.set_title("simulator-corrected MILP convergence")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{prefix}fig_convergence.png", dpi=160)
            print(f"-> {prefix}fig_convergence.png")
    except Exception as e:
        print(f"(figures skipped: {e})")


def main():
    from run_tile import make_synthetic_users
    hex_id, lat, lon, mbps = make_synthetic_users()
    inst = build_instance(lat, lon, mbps, hex_id,
                          rho_cand=1.0, agg_res=9, K_elig=6)
    ux, uy = project_km(lat, lon, inst.lat0)
    user_xy = np.column_stack([ux, uy])

    rows, hist = run_all(inst, user_xy, se_default)
    print("\n=== SUMMARY ===")
    print(f"{'method':14s} {'obj':>12s} {'served%':>8s} {'sites':>6s} "
          f"{'beams':>6s} {'wall_s':>7s}")
    for r in rows:
        print(f"{r['method']:14s} {r['objective']:12.0f} "
              f"{r['served_pct']:8.2f} {r['sites']:6d} {r['beams']:6d} "
              f"{r['wall_s']:7.1f}")
    save_outputs(rows, hist)


if __name__ == "__main__":
    main()