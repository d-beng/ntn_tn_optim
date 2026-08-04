#!/usr/bin/env python3
"""
drop_targeted_loop.py — drive the SIMULATOR's measured drops toward zero.

The inner MILP can only guarantee zero outage in ITS OWN model. The simulator
drops users for reasons the deterministic model cannot represent:
  * log-normal shadowing (a user in a deep fade drops however good the plan is)
  * within-demand-point SE spread (one eta per aggregation hex, many users in it)
  * greedy per-user attachment instead of genie fractional association
  * mobility between the planning snapshot and the simulated hour
So "zero drops" is not a constraint you can write down -- it is a target you
converge to, by provisioning margin exactly where the simulator loses users.

MECHANISM (scenario-cut flavoured, not a global safety factor):
    repeat:
        solve MILP  (demand provisioned at d_u * s_u)
        run the real simulator on the resulting network
        measure drops PER DEMAND POINT
        if total drops <= target: stop
        s_u <- s_u * (1 + beta * drop_fraction_u)   for the points that dropped

FIXES IN THIS REVISION
----------------------
* lam / c_beam defaults were 100 / 5 -- placeholders from the earliest
  prototype. Any caller that forgot to pass them (smoke_real_sim.py did) was
  silently optimising a COMPLETELY DIFFERENT cost model from the rest of the
  project. They now default to the literature-grounded 1309 / 49813 imported
  from hex_milp, so there is exactly one source of truth.
* The loop returned the LAST iterate even when an earlier one was better.
  It now tracks and returns the BEST iterate by measured simulator drops, and
  reports which iteration that was.
* Guards against a solve that returns no incumbent.

Convergence is empirical (as in all simulation-optimisation of this shape):
report the drop trajectory, the residual ||s_{k+1} - s_k||, and a sweep over
beta / se_percentile showing the undamped setting oscillates.
"""
from __future__ import annotations
import time
from typing import Callable, Dict, List

import numpy as np

from hex_milp import solve_hex, LAM_DEFAULT, C_BEAM_DEFAULT


def solve_to_drop_target(inst,
                         sim_fn: Callable,       # y, x_assign -> (eta, rho, drop_by_dem)
                         target_drop_mbps: float = 10.0,
                         max_iter: int = 8,
                         beta: float = 1.0,      # margin aggressiveness
                         use_divergence: bool = True,
                         force_serve: bool = False,     # o_u=0 at drop points
                         patience: int = 2,      # stop after N worsening iters
                         damping: float = 0.5,   # on eta / rho
                         max_scale: float = 4.0, # cap on provisioning margin
                         require_coverage: bool = False,
                         # measured on hex 852b9bd7 with literature costs:
                         # forcing >=1 site per demand point moved sites to
                         # isolated points (0.39 Mbps each, below the 2.24 Mbps
                         # a UMi site needs to pay for itself) and made the
                         # simulator's drops WORSE: 25.2 -> 43.2 Gbps.
                         lam: float = LAM_DEFAULT,
                         c_beam: float = C_BEAM_DEFAULT,
                         mip_gap: float = 0.02, time_limit_s: float = 600.0,
                         threads: int = 0, log: bool = True):
    U = len(inst.dem_mbps)
    scale = np.ones(U)
    cap_factor: Dict[int, float] = {}
    history: List[dict] = []
    t0 = time.time()
    res = None
    best = None            # (sim_drop, iter, result, scale copy)

    for it in range(1, max_iter + 1):
        fs = None
        if force_serve and history:
            fs = [u for u, v in (history[-1].get("drop_by_dem") or {}).items()
                  if v > 1e-9]
        res = solve_hex(inst, lam=lam, c_beam=c_beam, mip_gap=mip_gap,
                        time_limit_s=time_limit_s, threads=threads, log=False,
                        require_coverage=require_coverage,
                        demand_scale=scale, force_serve=fs,
                        cap_factor=cap_factor or None)
        if (res["status"].startswith("Infeasible") or res.get("no_solution")) \
                and fs:
            if log:
                print(f"  [drop-loop {it}] force_serve INFEASIBLE -> those "
                      f"demand points cannot be served on this candidate set; "
                      f"retrying without the hard constraint (use the "
                      f"lexicographic solver for the certified floor)",
                      flush=True)
            res = solve_hex(inst, lam=lam, c_beam=c_beam, mip_gap=mip_gap,
                            time_limit_s=time_limit_s, threads=threads,
                            log=False, require_coverage=require_coverage,
                            demand_scale=scale, cap_factor=cap_factor or None)
        n_sites = int(sum(res["opened"].values()))
        if n_sites == 0:
            if log:
                print(f"  [drop-loop {it}] solver returned ZERO sites "
                      f"({res['status']}, gap={res['gap']:.3f}). That is a "
                      f"solver artifact at lam={lam:,.0f}, not economics. "
                      f"Stopping.", flush=True)
            break

        eta_real, rho, drop_by_dem = sim_fn(res["y"], res.get("x_assign"))
        # ATTACHMENT-AWARE CAPACITY. Measured on the real hex: users attach up
        # to 2.9x more heavily than the plan assigns them. Capacity added AT a
        # dropping demand point is therefore not where those users actually
        # attach -- it sits idle while the popular cell still congests. Feeding
        # the divergence delta_j = attached/assigned into the derating tells
        # the model "site j effectively has W/delta_j usable", which makes it
        # build RELIEF NEAR the over-subscribed site instead.
        divergence = getattr(sim_fn, "last_divergence", None) or {}
        total_drop = float(sum(drop_by_dem.values())) if drop_by_dem else 0.0
        n_bad = sum(1 for v in (drop_by_dem or {}).values() if v > 1e-9)

        if best is None or total_drop < best[0] - 1e-6:
            best = (total_drop, it, res, scale.copy())

        history.append({
            "iter": it, "sites": n_sites, "beams": res["beams"],
            "status": res["status"], "gap": res["gap"],
            "model_outage_mbps": res["outage_mbps"],
            "sim_drop_mbps": total_drop, "dem_pts_dropping": n_bad,
            "mean_scale": float(scale.mean()), "max_scale": float(scale.max()),
            "mean_divergence": (float(np.mean(list(divergence.values())))
                                if divergence else None),
            "drop_by_dem": drop_by_dem,
            "wall_s": time.time() - t0,
        })
        if log:
            print(f"  [drop-loop {it}] sites={n_sites:,} "
                  f"model_outage={res['outage_mbps']:,.0f} Mbps  "
                  f"SIM_DROP={total_drop:,.0f} Mbps at {n_bad:,} demand pts  "
                  f"margin mean={scale.mean():.2f} max={scale.max():.2f}"
                  + (f"  div={np.mean(list(divergence.values())):.2f}"
                     if divergence else ""),
                  flush=True)

        # divergence guard: if the simulator's drops are getting WORSE, the
        # instrument is not working on this instance. Say so and stop rather
        # than burning long simulations making it worse.
        if len(history) >= 2:
            worse = sum(1 for a, b in zip(history[-patience-1:], history[-patience:])
                        if b["sim_drop_mbps"] > a["sim_drop_mbps"] + 1e-6)
            if worse >= patience:
                if log:
                    dv = (np.mean(list(divergence.values()))
                          if divergence else float("nan"))
                    print(f"  STOPPING: simulator drops rose {patience}x in a "
                          f"row ({[round(h['sim_drop_mbps']) for h in history]})."
                          f" Margin is the wrong instrument here -- the census "
                          f"says congestion, and divergence is {dv:.2f}. "
                          f"Fix association (steering) or add spectrum, not "
                          f"provisioning margin.", flush=True)
                break

        if total_drop <= target_drop_mbps:
            if log:
                print(f"  TARGET MET: simulator drops {total_drop:,.1f} Mbps "
                      f"<= {target_drop_mbps:,.1f}", flush=True)
            break
        if it == max_iter:
            if log:
                print(f"  max_iter reached; simulator still drops "
                      f"{total_drop:,.1f} Mbps. Lower se_percentile, raise "
                      f"beta/max_scale, or the target is infeasible for this "
                      f"candidate set (check no_eligible in diagnose_outage).",
                      flush=True)
            break

        # ---- localized margin: over-provision ONLY where users were lost ----
        for u, lost in (drop_by_dem or {}).items():
            if lost <= 1e-9:
                continue
            d = float(inst.dem_mbps[u])
            frac = min(1.0, lost / max(d, 1e-9))
            scale[u] = min(max_scale, scale[u] * (1.0 + beta * frac))

        # ---- physics feedback, damped ----
        for u in range(U):
            inst.elig_se[u] = [(1 - damping) * a + damping * b
                               for a, b in zip(inst.elig_se[u], eta_real[u])]
        for j, r in (rho or {}).items():
            eff = float(r)
            if use_divergence:
                dj = float(divergence.get(j, 1.0))
                if dj > 1.0:                       # over-subscribed vs plan
                    eff = eff / dj
            eff = min(1.0, max(0.05, eff))
            cap_factor[j] = (1 - damping) * cap_factor.get(j, 1.0) + damping * eff

    # ---- return the BEST iterate, not the last one ------------------------
    if best is not None:
        best_drop, best_it, best_res, best_scale = best
        if log and history and best_it != history[-1]["iter"]:
            print(f"  RETURNING iteration {best_it} (best measured drops "
                  f"{best_drop:,.0f} Mbps), not the last one "
                  f"({history[-1]['sim_drop_mbps']:,.0f} Mbps).", flush=True)
        res = best_res
        res["demand_scale"] = best_scale
        res["best_iter"] = best_it
        res["best_sim_drop_mbps"] = best_drop
    else:
        res = res or {}
        res["demand_scale"] = scale
    res["history"] = history
    return res, history