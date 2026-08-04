#!/usr/bin/env python3
"""
demand_field.py — turn a PROBABILISTIC mobility model into deterministic
MILP data, without growing the model.

THE PRINCIPLE
-------------
The MILP never sees users; it sees occupancy MASS aggregated to H3 cells.
A user with a home and several attractors therefore does not need a "vector"
in the model -- it contributes fractional mass to several cells:

    d_{c,t} = sum_u pi_u(c,t) * d_u(t)

Because pi is random, d_{c,t} is a RANDOM VARIABLE. Planning against its mean
is the same error as planning against median SE: half the realisations are
worse and drop. So we carry the second moment as well, estimate it by
replicating the mobility model, and hand the optimiser a QUANTILE field.

The MILP stays deterministic, linear, and exactly the same size. All the
uncertainty is spent here, in preprocessing, where it costs O(R) passes over
the users instead of O(R) copies of the model.

WHY REPLICATION RATHER THAN A CLOSED FORM
-----------------------------------------
If every user moved independently you could write
    E[D_c]   = sum_u p_uc d_u
    Var[D_c] = sum_u p_uc (1-p_uc) d_u^2
in one pass. But real mobility models have shared structure: users sharing an
attractor are positively correlated, and a user cannot be in two cells at
once (negative correlation across cells for the same user). Replicating the
model captures whatever correlation it actually has, including any you did
not intend. It is also robust to the mobility model changing later.

WHERE THE VARIANCE BINDS  (this is the non-obvious part)
--------------------------------------------------------
The capacity constraint is per (site, sector), and a sector's load is a
random sum over the ~300 users that land in it. How those users are
PARTITIONED into demand points does not change their sum. Measured on the
GTA instance:

    demand point, res-10 :  27 users -> CV 0.27  -> +81% at z=3
    demand point, res-11 : 6.8 users -> CV 0.54  -> +162% at z=3
    (site, sector)       : 297 users -> CV 0.082 -> +25% at z=3

So applying a quantile PER DEMAND POINT over-provisions by 3-6x. The right
place is the site-sector aggregate, via the cap_factor hook that already
exists in hex_milp.solve_hex(). Use site_derating() below for that, and use
the per-point mean field for the demand itself.

Corollary worth stating in the paper: aggregation resolution controls
COVERAGE error only; demand stochasticity is a site-level quantity that is
invariant to resolution. The two knobs are orthogonal.

And: CV grows as 1/sqrt(n), so the zero-drop risk lives in SPARSE sectors
(20 users -> +95% margin), not in the dense urban ones. That is the opposite
of where a drop-triggered margin loop puts capacity.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
def sample_demand_field(users, hours: Sequence[float], agg_res: int = 10,
                        n_rep: int = 20, seed: int = 0,
                        move_fn: Callable | None = None,
                        demand_fn: Callable | None = None,
                        log: bool = True):
    """Replicate the mobility model and accumulate per-cell demand moments.

    users     : the User objects (mutated in place by move(); that is fine,
                each replication overwrites the previous position).
    hours     : representative periods, e.g. (8, 12, 20). ONE build serves
                all of them; only the association variables multiply.
    n_rep     : replications per period. 20 is plenty for a mean/variance --
                the standard error of the mean is CV/sqrt(20) ~ 2% of the
                mean at CV 0.08.
    move_fn   : (user, hour, rep) -> None. Defaults to user.move(hour, 5).
                Override if your mobility model needs an explicit seed to
                produce DIFFERENT draws per replication -- if move() is
                deterministic given the hour, every replication is identical
                and the variance comes out zero (the function warns).
    demand_fn : (user, hour) -> Mbps. Defaults to get_demand_at_time(hour).

    Returns (cells, E, V, N) where
        cells : list of H3 ids, length C
        E     : (C, T) mean demand [Mbps]
        V     : (C, T) variance of demand [Mbps^2]
        N     : (C, T) mean user count (used for sparse-sector diagnostics)
    """
    import h3
    move_fn = move_fn or (lambda u, hr, rep: u.move(hr, 5))
    demand_fn = demand_fn or (lambda u, hr: u.get_demand_at_time(hr))
    rng = np.random.default_rng(seed)

    T = len(hours)
    # accumulate sum and sum-of-squares ACROSS REPLICATIONS, per (cell, period)
    s1 = defaultdict(float)
    s2 = defaultdict(float)
    cnt = defaultdict(float)
    for t_i, hr in enumerate(hours):
        for rep in range(n_rep):
            per_cell = defaultdict(float)
            per_cell_n = defaultdict(int)
            for u in users:
                move_fn(u, hr, rep)
                c = h3.latlng_to_cell(float(u.current_lat),
                                      float(u.current_lon), agg_res)
                per_cell[c] += float(demand_fn(u, hr))
                per_cell_n[c] += 1
            for c, v in per_cell.items():
                s1[(c, t_i)] += v
                s2[(c, t_i)] += v * v
                cnt[(c, t_i)] += per_cell_n[c]
            if log:
                print(f"    [field] hour {hr:g} rep {rep+1}/{n_rep}: "
                      f"{len(per_cell):,} occupied cells", flush=True)

    cells = sorted({c for c, _ in s1})
    idx = {c: i for i, c in enumerate(cells)}
    E = np.zeros((len(cells), T))
    V = np.zeros((len(cells), T))
    N = np.zeros((len(cells), T))
    for (c, t_i), tot in s1.items():
        i = idx[c]
        m = tot / n_rep
        E[i, t_i] = m
        # population variance across replications (biased by 1/n; with n_rep
        # >= 20 the difference is immaterial and the biased form is stable)
        V[i, t_i] = max(s2[(c, t_i)] / n_rep - m * m, 0.0)
        N[i, t_i] = cnt[(c, t_i)] / n_rep

    if log:
        tot_cv = np.sqrt(V.sum(axis=0)) / np.maximum(E.sum(axis=0), 1e-9)
        print(f"    [field] {len(cells):,} cells x {T} periods; "
              f"aggregate CV per period: "
              f"{', '.join(f'{c:.4f}' for c in tot_cv)}", flush=True)
        if V.max() <= 0:
            print("    [field] !! VARIANCE IS ZERO EVERYWHERE. Your move_fn is "
                  "deterministic given the hour, so every replication produced "
                  "the same positions. Pass a move_fn that varies with `rep` "
                  "(e.g. reseeds the user's RNG) or the chance constraint is "
                  "vacuous.", flush=True)
    return cells, E, V, N


# ---------------------------------------------------------------------------
def site_derating(inst, x_assign, V_by_point, z: float = 3.0,
                  floor: float = 0.25):
    """Per-site capacity derating implementing a chance constraint.

    The exact form of
        P( sum_u (D_u/eta_uj) x_uj  <=  W rho_j y_j ) >= 1 - eps
    is second-order conic and would break the MILP. Instead we solve with the
    MEAN field, read off which demand points landed on each site, and derate
    that site's usable bandwidth by the aggregate coefficient of variation:

        Delta_j = z * sqrt( sum_{u->j} V_u ) / sum_{u->j} E_u
        rho_j  <- rho_j / (1 + Delta_j)

    Two or three passes converge; no simulation is needed. This is the same
    `cap_factor` hook solve_hex() already takes for the simulator's rho, so
    nothing new is plumbed -- multiply the two together when using both.

    Because CV ~ 1/sqrt(n), this automatically puts the margin in SPARSE
    sectors, which is where zero-drop actually fails, instead of wherever the
    last simulation happened to lose users.
    """
    E_by_j = defaultdict(float)
    V_by_j = defaultdict(float)
    for (u, j), frac in (x_assign or {}).items():
        if frac <= 1e-9:
            continue
        E_by_j[j] += frac * float(inst.dem_mbps[u])
        # variance of a scaled sum: fraction frac of the point's users
        V_by_j[j] += frac * float(V_by_point[u])
    cap = {}
    for j, e in E_by_j.items():
        if e <= 1e-9:
            continue
        delta = z * math.sqrt(max(V_by_j[j], 0.0)) / e
        cap[int(j)] = max(floor, 1.0 / (1.0 + delta))
    return cap


# ---------------------------------------------------------------------------
def build_multiperiod_demand(cells, E, V, quantile_z: float = 0.0):
    """Collapse the (cell, period) field into what build_instance consumes.

    quantile_z = 0.0 -> plan on the MEAN and put the margin at site level via
                        site_derating() (RECOMMENDED: correct place, ~25%).
    quantile_z > 0.0 -> plan on E + z*sqrt(V) per demand point. Simple, but
                        over-provisions 3-6x because it applies a site-level
                        statistic at demand-point granularity. Provided only
                        so the two can be compared in an ablation.

    Do NOT collapse periods with max_t: that pays for the downtown noon peak
    AND the suburban evening peak at every location, as if they were
    simultaneous. Keep the period index and let one y serve all of them.
    """
    D = E + quantile_z * np.sqrt(np.maximum(V, 0.0))
    return D


def peak_collapse_penalty(E):
    """Diagnostic: how much a max-over-periods collapse would over-build.
    Returns (sum_c max_t E_ct) / (max_t sum_c E_ct) -- 1.0 means demand does
    not move, larger means it does and the multi-period model is earning its
    keep. This single number justifies the extra periods in the paper."""
    return float(E.max(axis=1).sum() / max(E.sum(axis=0).max(), 1e-9))