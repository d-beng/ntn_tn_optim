#!/usr/bin/env python3
"""
risk_solve.py — the link between demand uncertainty and the existing MILP.

WHAT THIS IS
------------
The chance constraint

    P( sum_u (D_u / eta_uj) x_uj  <=  W_t rho_j y_j )  >=  1 - eps

is second-order conic and would break the MILP. This module implements it
the cheap way: solve with the MEAN demand field, read off which demand points
actually landed on each site, convert that site's aggregate coefficient of
variation into a bandwidth derating, and re-solve. Two or three passes
converge and NO simulation is involved.

    Delta_j = z * sqrt( sum_{u->j} V_u ) / sum_{u->j} E_u
    rho_j  <- rho_j / (1 + Delta_j)

The derating goes through `cap_factor`, which solve_hex() already accepts for
the simulator's measured rho. Nothing new is plumbed: the two multiply.

WHY AT SITE LEVEL AND NOT PER DEMAND POINT
------------------------------------------
The binding constraint is per (site, sector) and a sector's load is a random
sum over the ~300 users that land in it. Partitioning those users into finer
demand points does not change their sum. Measured on the GTA instance:

    per demand point, res-10 :  27 users -> CV 0.27  -> +81%  at z=3
    per demand point, res-11 : 6.8 users -> CV 0.54  -> +162% at z=3
    per (site, sector)       : 297 users -> CV 0.082 -> +25%  at z=3

Applying the quantile per demand point over-provisions by 3-6x. It also
double-counts against aggregation resolution, which controls COVERAGE error
and nothing else. The two knobs are orthogonal.

Because CV ~ 1/sqrt(n), this puts margin in SPARSE sectors (20 users -> +95%)
rather than dense ones -- which is where zero-drop actually fails, and the
opposite of where a drop-triggered margin loop puts capacity.

WHERE THE VARIANCE COMES FROM
-----------------------------
Preferred : demand_field.sample_demand_field() replicates the mobility model
            and measures it, capturing whatever correlation the model has.
Fallback  : inst.dem_var, accumulated for free by build_instance as
            sum_u d_u^2 over the users in each cell -- the compound-Poisson
            variance of that cell's load. Requires no stochastic mobility and
            no extra pass. Use this to get started; swap in the measured
            field when the mobility model is confirmed stochastic.
"""
from __future__ import annotations
import math
from collections import defaultdict

import numpy as np

from hex_milp import solve_hex, LAM_DEFAULT, C_BEAM_DEFAULT


# ---------------------------------------------------------------------------
def variance_by_point(inst, measured=None, cv_user: float | None = None):
    """Per-demand-point variance of offered load [Mbps^2].

    measured : optional array aligned to inst.dem_mbps (from demand_field).
    cv_user  : if given, rescale the analytic estimate for per-user demand
               dispersion other than the compound-Poisson default.
    """
    U = len(inst.dem_mbps)
    if measured is not None:
        V = np.asarray(measured, dtype=float).reshape(-1)[:U]
        if V.size != U:
            raise ValueError(f"measured variance has {V.size} entries, "
                             f"instance has {U} demand points")
        return V
    dv = getattr(inst, "dem_var", ())
    if dv is not None and len(dv) == U:
        V = np.asarray(dv, dtype=float)          # sum_u d_u^2 per cell
        if cv_user is not None:
            n = np.asarray(getattr(inst, "dem_n", np.ones(U)), dtype=float)
            dbar = inst.dem_mbps / np.maximum(n, 1.0)
            V = n * dbar ** 2 * (1.0 + cv_user ** 2)
        return V
    # last resort: assume Poisson occupancy with a single mean user demand
    n = np.asarray(getattr(inst, "dem_n", None) or
                   np.ones(U) * 25.0, dtype=float)
    dbar = inst.dem_mbps / np.maximum(n, 1.0)
    return n * dbar ** 2 * 2.0                   # CV_d = 1 => E[d^2] = 2 dbar^2


def site_derating(inst, x_assign, V, z: float = 3.0, floor: float = 0.25):
    """cap_factor dict implementing the chance constraint at site level."""
    E_j = defaultdict(float)
    V_j = defaultdict(float)
    for (u, j), frac in (x_assign or {}).items():
        if frac <= 1e-9:
            continue
        E_j[j] += frac * float(inst.dem_mbps[u])
        V_j[j] += frac * float(V[u])
    cap = {}
    for j, e in E_j.items():
        if e <= 1e-9:
            continue
        delta = z * math.sqrt(max(V_j[j], 0.0)) / e
        cap[int(j)] = max(floor, 1.0 / (1.0 + delta))
    return cap


def merge_cap(*caps):
    """Combine independent deratings multiplicatively (risk margin x measured
    rho). Missing keys default to 1.0."""
    out = {}
    for c in caps:
        for j, v in (c or {}).items():
            out[j] = out.get(j, 1.0) * float(v)
    return {j: max(0.05, min(1.0, v)) for j, v in out.items()}


# ---------------------------------------------------------------------------
def solve_with_demand_risk(inst, z: float = 3.0, n_passes: int = 3,
                           measured_var=None, sim_cap=None,
                           tol: float = 0.02, log: bool = True,
                           lam: float = LAM_DEFAULT,
                           c_beam: float = C_BEAM_DEFAULT, **solve_kw):
    """Solve the MILP under a per-site chance constraint on demand.

    sim_cap : the simulator's measured rho, if you have it. Multiplied in.
    Returns the final result dict, with 'risk_cap' and 'risk_passes' added.
    """
    V = variance_by_point(inst, measured=measured_var)
    if log:
        cv = math.sqrt(float(V.sum())) / max(float(inst.dem_mbps.sum()), 1e-9)
        print(f"  [risk] aggregate CV of offered load {cv:.4f}; "
              f"planning at z={z:.1f} ({100*(1-0.5*math.erfc(z/math.sqrt(2))):.3f}% "
              f"per-site service level)", flush=True)

    cap_risk = None
    res = None
    for p in range(1, n_passes + 1):
        cap = merge_cap(cap_risk, sim_cap) if (cap_risk or sim_cap) else None
        res = solve_hex(inst, lam=lam, c_beam=c_beam, cap_factor=cap,
                        log=False, **solve_kw)
        if res.get("no_solution") or not res["y"].any():
            if log:
                print(f"  [risk] pass {p}: no placement ({res['status']})",
                      flush=True)
            break
        new = site_derating(inst, res["x_assign"], V, z=z)
        n_sites = int(sum(res["opened"].values()))
        if new:
            arr = np.array(list(new.values()))
            if log:
                print(f"  [risk] pass {p}: {n_sites:,} sites, "
                      f"served={res['served_pct']:.2f}%, derating mean="
                      f"{arr.mean():.3f} min={arr.min():.3f} "
                      f"(margin +{100*(1/arr.mean()-1):.1f}%)", flush=True)
        if cap_risk is not None and new:
            shared = set(cap_risk) & set(new)
            if shared:
                drift = max(abs(cap_risk[j] - new[j]) for j in shared)
                if drift < tol:
                    if log:
                        print(f"  [risk] converged (max drift {drift:.4f} "
                              f"< {tol})", flush=True)
                    cap_risk = new
                    break
        cap_risk = new

    if res is not None:
        res["risk_cap"] = cap_risk
        res["risk_passes"] = p
    return res


# ---------------------------------------------------------------------------
def sparse_sector_report(inst, res, V, z: float = 3.0, top: int = 10,
                         log: bool = True):
    """Which sites carry the most demand risk. These are where zero-drop
    fails first, and they are usually NOT the congested urban ones."""
    E_j = defaultdict(float)
    V_j = defaultdict(float)
    for (u, j), frac in (res.get("x_assign") or {}).items():
        if frac <= 1e-9:
            continue
        E_j[j] += frac * float(inst.dem_mbps[u])
        V_j[j] += frac * float(V[u])
    rows = []
    for j, e in E_j.items():
        if e <= 1e-9:
            continue
        cv = math.sqrt(max(V_j[j], 0.0)) / e
        rows.append((cv, int(j), e, inst.tiers[inst.cand_tier[j]].name))
    rows.sort(reverse=True)
    if log and rows:
        print(f"  [risk] highest-CV sites (margin needed at z={z:.1f}):",
              flush=True)
        for cv, j, e, tname in rows[:top]:
            print(f"        cand {j:6d} {tname:8s} load {e:8.1f} Mbps  "
                  f"CV={cv:.3f} -> +{100*z*cv:5.1f}%", flush=True)
        allcv = np.array([r[0] for r in rows])
        print(f"        median CV {np.median(allcv):.3f}, "
              f"p90 {np.percentile(allcv, 90):.3f} over {len(rows)} sites",
              flush=True)
    return rows