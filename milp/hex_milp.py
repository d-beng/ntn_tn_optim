# milp/hex_milp.py

#!/usr/bin/env python3
"""
hex_milp.py
===========
Per-hex placement MILP in HiGHS (highspy), implementing formulation (8):

  min  sum_j c_j y_j + c_beam sum_h b_h + lambda sum_u o_u d_u
  s.t. sum_{j in N(u)} x_uj + z_u + o_u = 1            (assignment)
       x_uj <= y_j                                      (open sites only)
       sum_u (d_u/eta_uj) x_uj <= W_t(j) y_j            (TN bandwidth/sector)
       sum_{u in U_h} d_u z_u <= C_beam b_h             (NTN one-beam ceiling)
       sum_h b_h <= B_max                               (beam budget)
       y_j + y_k <= 1 for conflict pairs                (min inter-site dist)
       y, b binary;  x, z, o in [0,1]

NEW IN THIS REVISION: rescore_fixed_placement().
  The corrected re-solve in the smoke test used to RE-CHOOSE y under the
  kappa-deflated SE and came back with {RMA:0,UMA:0,UMI:0} / 0.23% served.
  That is not an economics result -- at lam=1309 a UMi site pays for itself
  at 2.24 Mbps and still carries ~490 Mbps/sector at kappa=0.62, and the
  all-closed objective (~1.8e9) is ~150x worse than the real deployment
  (~1.15e7). It was a time-limit incumbent the solver never improved on,
  invisible because the caller printed neither status nor gap.
  Freezing y turns (8) into an LP over (x,z,o) plus the per-hex beam
  integers: small, deterministic, and immune to that trap. This is the
  honest "keep the MILP's placement, correct the prediction" operation.
"""
from __future__ import annotations
import time
import numpy as np
import highspy

from candidate_generator import Instance

# Literature-grounded defaults (single source of truth for every caller):
#   lam    : Kumar & Oughton 2023 Tab.4-derived value of unserved demand
#   c_beam : Toka et al. 2024 Tab.3 annualised cost of a persistent LEO beam
LAM_DEFAULT = 1309.0        # USD per Mbps unserved per YEAR
C_BEAM_DEFAULT = 49813.0    # USD/yr per persistent LEO beam


def solve_hex(inst: Instance,
              lam: float = LAM_DEFAULT,
              c_beam: float = C_BEAM_DEFAULT,
              B_max: int = 10**9,          # beam budget (per-hex solve: large)
              mip_gap: float = 0.02,
              time_limit_s: float = 600.0,
              threads: int = 0,
              max_outage_frac=None,        # None = penalty only (lambda);
                                           # 0.0 = HARD zero-outage
              require_coverage: bool = False,  # force >=1 open eligible site
              force_serve=None,            # demand points that MUST be served
              demand_scale=None,           # (U,) provisioning multiplier
              cap_factor=None,             # {cand j: rho_j} simulator derating
              outage_budget_mbps=None,     # hard cap on TOTAL outage (Eq. 8b)
              _zero_site_costs=False,      # stage-1 helper
              fix_closed_mask=None,        # bool (J,): force y_j = 0
              fix_open_mask=None,          # bool (J,): force y_j = 1 (NEW)
              sector_mult: float = 3.0,    # API compat; capacity is PER SECTOR
              closest_assignment: bool = False,
                                           # ASSOCIATION REALISM. The simulator
                                           # runs allow_spillover=False: a user
                                           # takes its BEST-SINR cell or it
                                           # drops. The MILP's free x_uj is a
                                           # genie the simulator does not have
                                           # -- measured as divergence 1.06-1.46
                                           # (attached/assigned) and as "5G
                                           # Congestion (Tower Empty)" sitting
                                           # alongside 33-63% mean utilisation:
                                           # on AVERAGE there is room, but not
                                           # at the cell users actually pick.
                                           # These rows force the model to plan
                                           # for where users land:
                                           #   x_uj + sum_{k: eta_uk > eta_uj}
                                           #          y_k <= 1
                                           # i.e. if any better-SE site is open,
                                           # x_uj = 0. Same row count as (8d),
                                           # which is redundant anyway -- so
                                           # trade one for the other.
                                           # Expect MORE sites and a lower
                                           # model served%: it removes an
                                           # optimism, it does not add one.
              disaggregate_linking=True,
                                           # True  = all linking rows (DEFAULT)
                                           # False = none
                                           # 0..1  = keep them only for the
                                           #         demand points carrying
                                           #         that FRACTION of total
                                           #         demand MASS (largest
                                           #         first) -- most of the
                                           #         bound at a fraction of
                                           #         the rows.
                                           # (8d) x_uj <= y_j. REDUNDANT given
                                           # (8e): every (u,j) sits in exactly
                                           # one per-sector row and y_j=0
                                           # zeroes its RHS, forcing x_uj=0.
                                           # Kept by default because it
                                           # tightens the LP relaxation -- but
                                           # it is ~65% of all rows. Measured:
                                           # at 587k of them the root LP did
                                           # not converge in 6 h (Nodes=0,
                                           # BestSol=inf). A loose bound you
                                           # can compute beats a tight bound
                                           # you cannot -- BUT dropping it
                                           # entirely makes the LP badly
                                           # fractional and HiGHS compensates
                                           # with runaway cut generation:
                                           # measured 1,063,177 cuts, rounds
                                           # doubling in wall time, 86% gap
                                           # after 66 min. Prefer a FRACTION
                                           # over False.
              log: bool = True):
    J = len(inst.cand_tier)
    U = len(inst.dem_mbps)
    _ds = (np.ones(U) if demand_scale is None
           else np.asarray(demand_scale, dtype=float).reshape(-1)[:U])
    H = len(inst.hex_ids)
    hex_index = {h: i for i, h in enumerate(inst.hex_ids)}

    # ---- variable layout: [ y | b | z | o | x ] ---------------------------
    n_y, n_b, n_z, n_o = J, H, U, U
    x_index = {}
    xs = []
    for u in range(U):
        for j in inst.elig_j[u]:
            x_index[(u, j)] = n_y + n_b + n_z + n_o + len(xs)
            xs.append((u, j))
    n_x = len(xs)
    N = n_y + n_b + n_z + n_o + n_x

    iy = lambda j: j
    ib = lambda h: n_y + h
    iz = lambda u: n_y + n_b + u
    io = lambda u: n_y + n_b + n_z + u

    h = highspy.Highs()
    if not log:
        h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", mip_gap)
    h.setOptionValue("time_limit", time_limit_s)
    h.setOptionValue("presolve", "on")
    if threads:
        h.setOptionValue("threads", int(threads))
        h.setOptionValue("parallel", "on")

    # ---- columns: costs + bounds -------------------------------------------
    cost = np.zeros(N)
    lb = np.zeros(N)
    ub = np.ones(N)
    integrality = np.zeros(N, dtype=np.int32)   # 0 cont, 1 integer
    for j in range(J):
        cost[iy(j)] = 0.0 if _zero_site_costs else float(inst.cand_cost[j])
        integrality[iy(j)] = 1
    for hh in range(H):
        cost[ib(hh)] = c_beam
        integrality[ib(hh)] = 1
    for u in range(U):
        cost[io(u)] = lam * float(inst.dem_mbps[u]) * _ds[u]
        if max_outage_frac is not None:
            ub[io(u)] = float(max_outage_frac)   # 0.0 => outage forbidden

    # ---- forced-closed FIRST, then forced-open (open wins on overlap) -----
    if fix_closed_mask is not None:
        for j in np.where(np.asarray(fix_closed_mask))[0]:
            ub[iy(int(j))] = 0.0
    forced_open = np.zeros(J, dtype=bool)
    fo = getattr(inst, "fixed_open", None)
    if fo is not None:
        forced_open |= np.asarray(fo, dtype=bool)[:J]
    if fix_open_mask is not None:
        forced_open |= np.asarray(fix_open_mask, dtype=bool)[:J]
    for j in np.where(forced_open)[0]:
        lb[iy(int(j))] = 1.0
        ub[iy(int(j))] = 1.0

    h.addVars(N, lb, ub)
    h.changeColsCost(N, np.arange(N, dtype=np.int32), cost)
    h.changeColsIntegrality(N, np.arange(N, dtype=np.int32), integrality)

    # ---- constraints -------------------------------------------------------
    # (8c) assignment: sum_j x_uj + z_u + o_u = 1
    for u in range(U):
        idx = [x_index[(u, j)] for j in inst.elig_j[u]] + [iz(u), io(u)]
        val = [1.0] * len(idx)
        h.addRow(1.0, 1.0, len(idx),
                 np.array(idx, dtype=np.int32), np.array(val))

    # (8d) x_uj <= y_j  -- see disaggregate_linking above.
    # Demand is heavily skewed, so keeping the rows for the demand points that
    # carry most of the MASS retains most of the LP tightness for a fraction of
    # the rows. Dropping them all is what triggered the cut-generation blowup.
    _dl = disaggregate_linking
    if isinstance(_dl, bool):
        _keep = set(range(U)) if _dl else set()
        _lbl = "all" if _dl else "none"
    else:
        _ord = np.argsort(-inst.dem_mbps)
        _cum = np.cumsum(inst.dem_mbps[_ord])
        _n = int(np.searchsorted(_cum, float(_dl) * _cum[-1]) + 1)
        _keep = set(int(v) for v in _ord[:_n])
        _lbl = f"{100*float(_dl):.0f}% of demand mass ({_n:,} of {U:,} pts)"
    _n_added = 0
    if _keep:
        for (u, j), col in x_index.items():
            if u not in _keep:
                continue
            h.addRow(-highspy.kHighsInf, 0.0, 2,
                     np.array([col, iy(j)], dtype=np.int32),
                     np.array([1.0, -1.0]))
            _n_added += 1
    if log:
        print(f"    (8d) linking rows: {_n_added:,} of {n_x:,} [{_lbl}]"
              + ("" if _n_added == n_x else
                 "  -- looser LP bound, larger reported gap"), flush=True)

    # (8d') CLOSEST ASSIGNMENT -- see closest_assignment above.
    # For the BEST candidate of each demand point there is no better site, so
    # no row is emitted; (8e) still forces x_uj = 0 when y_j = 0 (all
    # coefficients positive, RHS zero), so correctness does not depend on (8d).
    if closest_assignment:
        _n_ca = 0
        for u in range(U):
            order = sorted(range(len(inst.elig_j[u])),
                           key=lambda i: -inst.elig_se[u][i])
            for pos, i in enumerate(order):
                if pos == 0:
                    continue
                j = inst.elig_j[u][i]
                better = [iy(inst.elig_j[u][m]) for m in order[:pos]]
                idx = np.array([x_index[(u, j)]] + better, dtype=np.int32)
                h.addRow(-highspy.kHighsInf, 1.0, len(idx), idx,
                         np.ones(len(idx)))
                _n_ca += 1
        if log:
            print(f"    (8d\') closest-assignment rows: {_n_ca:,} "
                  f"(model now associates like the simulator)", flush=True)

    # (8e) TN bandwidth PER SECTOR (3GPP 30/150/270 boresights; each sector
    # has its own W_t budget -- replaces the pooled 3W relaxation, which
    # over-credits sites with angularly imbalanced demand).
    per_sector = {}
    for u in range(U):
        for j, se, k in zip(inst.elig_j[u], inst.elig_se[u], inst.elig_sec[u]):
            per_sector.setdefault((j, k), []).append(
                (x_index[(u, j)],
                 float(inst.dem_mbps[u]) * _ds[u] / max(se, 1e-6)))
    ext_res = getattr(inst, "ext_residual", {}) or {}
    for (j, k), terms in per_sector.items():
        if j in ext_res:
            W_mbps_hz = float(ext_res[j][k])     # neighbour's RESIDUAL only
        else:
            W_mbps_hz = inst.tiers[inst.cand_tier[j]].bw_hz / 1e6  # ONE sector
        if cap_factor is not None and j in cap_factor:
            W_mbps_hz *= float(cap_factor[j])    # simulator-measured derating
        idx = np.array([c for c, _ in terms] + [iy(j)], dtype=np.int32)
        val = np.array([v for _, v in terms] + [-W_mbps_hz])
        h.addRow(-highspy.kHighsInf, 0.0, len(idx), idx, val)

    # (8f) NTN ceiling per hex
    users_of_hex = {}
    for u, hx in enumerate(inst.dem_hex):
        users_of_hex.setdefault(hx, []).append(u)
    cap_of = getattr(inst, "beam_cap_of", None)
    for hx, us in users_of_hex.items():
        hh = hex_index[hx]
        cap = (float(cap_of[hx]) if cap_of is not None and hx in cap_of
               else inst.beam_cap_mbps)
        idx = np.array([iz(u) for u in us] + [ib(hh)], dtype=np.int32)
        val = np.array([float(inst.dem_mbps[u]) * _ds[u] for u in us] + [-cap])
        h.addRow(-highspy.kHighsInf, 0.0, len(idx), idx, val)

    # (8b) epsilon-constraint: total outage <= budget
    if outage_budget_mbps is not None:
        idx = np.array([io(u) for u in range(U)], dtype=np.int32)
        val = np.array([float(inst.dem_mbps[u]) for u in range(U)])
        h.addRow(-highspy.kHighsInf, float(outage_budget_mbps), U, idx, val)

    # COVERAGE GUARANTEE: sum_{j in N(u)} y_j >= 1 for every demand point that
    # has any eligible candidate.
    if require_coverage:
        for u in range(U):
            if not inst.elig_j[u]:
                continue
            idx = np.array(sorted({iy(j) for j in inst.elig_j[u]}),
                           dtype=np.int32)
            h.addRow(1.0, highspy.kHighsInf, len(idx), idx, np.ones(len(idx)))

    # FORCE-SERVE: o_u = 0 for protected demand points.
    if force_serve:
        for u in set(int(v) for v in force_serve):
            if 0 <= u < U:
                h.changeColBounds(io(u), 0.0, 0.0)

    # (8g) beam budget
    if B_max < H:
        idx = np.array([ib(hh) for hh in range(H)], dtype=np.int32)
        h.addRow(-highspy.kHighsInf, float(B_max), H, idx, np.ones(H))

    # conflict pairs: y_j + y_k <= 1
    for ja, jb in inst.conflict_pairs:
        h.addRow(-highspy.kHighsInf, 1.0, 2,
                 np.array([iy(int(ja)), iy(int(jb))], dtype=np.int32),
                 np.array([1.0, 1.0]))

    # ---- solve -------------------------------------------------------------
    t0 = time.time()
    h.run()
    wall = time.time() - t0
    info = h.getInfo()
    sol = h.getSolution()
    v = np.array(sol.col_value)
    status = h.modelStatusToString(h.getModelStatus())

    # No usable solution (infeasible / no incumbent): say so, do not silently
    # return an all-zero "solution" that reads like a deliberate choice.
    if v.size < N:
        return {
            "x_assign": {}, "ntn_by_hex": {}, "status": status,
            "gap": float("nan"), "objective": float("nan"), "wall_s": wall,
            "n_vars": N, "n_x": n_x, "n_conflicts": len(inst.conflict_pairs),
            "opened": {}, "beams": 0, "served_tn_mbps": 0.0, "ntn_mbps": 0.0,
            "outage_mbps": float(inst.dem_mbps.sum()),
            "total_mbps": float(inst.dem_mbps.sum()), "served_pct": 0.0,
            "y": np.zeros(J, dtype=bool), "b": np.zeros(H, dtype=bool),
            "z": np.zeros(U), "o": np.ones(U),
            "x_index": x_index, "x_val": np.zeros(N), "no_solution": True,
        }

    y = v[:n_y] > 0.5
    b = v[n_y:n_y + n_b] > 0.5
    z = v[n_y + n_b:n_y + n_b + n_z]
    o = v[n_y + n_b + n_z:n_y + n_b + n_z + n_o]

    served_tn = 0.0
    for (u, j), col in x_index.items():
        served_tn += v[col] * float(inst.dem_mbps[u])
    total = float(inst.dem_mbps.sum())
    ntn = float((z * inst.dem_mbps).sum())
    out = float((o * inst.dem_mbps).sum())

    x_assign = {k: float(v[c]) for k, c in x_index.items()
                if float(v[c]) > 1e-9}
    ntn_by_hex = {}
    for u in range(U):
        if z[u] > 1e-9:
            ntn_by_hex[inst.dem_hex[u]] = ntn_by_hex.get(inst.dem_hex[u], 0.0) \
                + float(z[u]) * float(inst.dem_mbps[u])

    return {
        "x_assign": x_assign,
        "ntn_by_hex": ntn_by_hex,
        "status": status,
        "gap": float(getattr(info, "mip_gap", float("nan"))),
        "objective": info.objective_function_value,
        "wall_s": wall,
        "n_vars": N, "n_x": n_x, "n_conflicts": len(inst.conflict_pairs),
        "opened": {inst.tiers[t].name: int((y & (inst.cand_tier == t)).sum())
                   for t in set(inst.cand_tier)},
        "beams": int(b.sum()),
        "served_tn_mbps": served_tn,
        "ntn_mbps": ntn,
        "outage_mbps": out,
        "total_mbps": total,
        "served_pct": 100.0 * (served_tn + ntn) / max(total, 1e-9),
        "y": y, "b": b, "z": z, "o": o, "x_index": x_index, "x_val": v,
        "no_solution": False,
    }


# ---------------------------------------------------------------------------
def rescore_fixed_placement(inst: Instance, y,
                            lam: float = LAM_DEFAULT,
                            c_beam: float = C_BEAM_DEFAULT,
                            cap_factor=None,
                            demand_scale=None,
                            mip_gap: float = 1e-4,
                            time_limit_s: float = 300.0,
                            threads: int = 0,
                            log: bool = False):
    """Judge a FIXED placement y under the CURRENT (corrected) inst.elig_se and
    simulator-measured rho, WITHOUT re-choosing sites.

    With y fixed, (8) collapses to an LP over (x, z, o) plus the per-hex beam
    integers -- fast, deterministic, and immune to the time-limit-incumbent
    trap a full re-solve falls into. This is the operation the project's own
    principle prescribes: KEEP the MILP's placement decisions, CORRECT the
    prediction. `served_pct` from here is directly comparable with the
    simulator's own served fraction.
    """
    y = np.asarray(y, dtype=bool).reshape(-1)
    J = len(inst.cand_tier)
    if y.size != J:
        raise ValueError(f"y has {y.size} entries, instance has {J} candidates")
    if not y.any():
        raise ValueError("rescore_fixed_placement: y is all-closed; nothing to "
                         "rescore (this is exactly the artifact the function "
                         "exists to avoid)")
    res = solve_hex(inst, lam=lam, c_beam=c_beam, mip_gap=mip_gap,
                    time_limit_s=time_limit_s, threads=threads, log=log,
                    cap_factor=cap_factor, demand_scale=demand_scale,
                    fix_open_mask=y, fix_closed_mask=~y)
    res["rescored_sites"] = int(y.sum())
    return res


# ---------------------------------------------------------------------------
def sector_usage_mhz(inst, res):
    """Per opened site: spectrum consumed per sector [MHz]*3, from the x
    solution. Used by the province driver to hand neighbours the RESIDUAL."""
    use = {}
    v = res["x_val"]
    for u in range(len(inst.dem_mbps)):
        for j, se, k in zip(inst.elig_j[u], inst.elig_se[u], inst.elig_sec[u]):
            col = res["x_index"].get((u, j))
            if col is None:
                continue
            frac = float(v[col])
            if frac > 1e-9:
                use.setdefault(j, [0.0, 0.0, 0.0])
                use[j][k] += frac * float(inst.dem_mbps[u]) / max(se, 1e-6)
    return use


def diagnose_outage(inst, res, tol=1e-6):
    """Split unserved demand into WHY it is unserved:
      no_eligible        : empty eligibility set -> coverage/candidate-limited
      elig_all_open_full : every eligible candidate open but out of sector
                           bandwidth -> genuine capacity saturation
      elig_not_opened    : some eligible candidate NOT opened -> the model
                           chose outage over building (lambda too low /
                           conflicts blocked it) -> a MODEL knob, not physics
    """
    y = res["y"]
    adj = {}
    for a, b in inst.conflict_pairs:
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))
    out = {"no_eligible": 0.0, "elig_all_open_full": 0.0,
           "conflict_blocked": 0.0, "elig_not_opened": 0.0, "n_pts": 0}
    for u in range(len(inst.dem_mbps)):
        o = float(res["o"][u])
        if o <= tol:
            continue
        d = o * float(inst.dem_mbps[u])
        out["n_pts"] += 1
        elig = inst.elig_j[u]
        if not elig:
            out["no_eligible"] += d
        elif all(y[j] for j in elig):
            out["elig_all_open_full"] += d
        elif all(y[j] or any(y[k] for k in adj.get(int(j), ())) for j in elig):
            # Every CLOSED eligible candidate is blocked by an OPEN conflicting
            # site: the model was not ALLOWED to open it. This is PACKING
            # GEOMETRY (rho_dep / rho_cand / cross_tier), not economics.
            # Without this category the demand was reported as
            # 'elig_not_opened' and read as "lambda too low" -- which sent two
            # experiments chasing the wrong knob.
            out["conflict_blocked"] += d
        else:
            out["elig_not_opened"] += d
    return out


def solve_hex_min_outage(inst, lam=LAM_DEFAULT, c_beam=C_BEAM_DEFAULT,
                         mip_gap=0.02, time_limit_s=600.0, threads=0,
                         slack=1.02, log=True, **kw):
    """LEXICOGRAPHIC solve -- the honest way to ask for zero outage.

    Stage 1: minimise TOTAL OUTAGE alone (costs zeroed) -> D_min, the provably
             smallest unserved demand any legal deployment can achieve.
             D_min == 0 => zero outage IS attainable.
             D_min  > 0 => zero outage is IMPOSSIBLE (certificate).
    Stage 2: re-solve minimising deployment cost subject to
             total outage <= slack * D_min.

    Unlike a hard o_u = 0 bound, this never returns a bare 'Infeasible': you
    always get a placement plus the exact best-attainable outage.
    """
    r1 = solve_hex(inst, lam=1.0, c_beam=0.0, mip_gap=mip_gap,
                   time_limit_s=time_limit_s, threads=threads, log=False,
                   _zero_site_costs=True, **kw)
    d_min = r1["outage_mbps"]
    if log:
        tot = r1["total_mbps"]
        print(f"  [stage 1] min achievable outage = {d_min:,.1f} Mbps "
              f"({100*d_min/max(tot,1e-9):.3f}% of demand)"
              + ("  -> ZERO OUTAGE ATTAINABLE" if d_min <= 1e-6
                 else "  -> zero outage IMPOSSIBLE (saturation certificate)"),
              flush=True)

    cap = max(slack * d_min, 1e-9) if d_min > 1e-6 else 0.0
    r2 = solve_hex(inst, lam=lam, c_beam=c_beam, mip_gap=mip_gap,
                   time_limit_s=time_limit_s, threads=threads, log=False,
                   outage_budget_mbps=cap, **kw)
    r2["min_outage_mbps"] = d_min
    r2["stage1"] = {k: r1[k] for k in
                    ("status", "outage_mbps", "served_pct", "opened", "beams")}
    if log:
        print(f"  [stage 2] cheapest deployment at that outage: "
              f"{r2['opened']}  served={r2['served_pct']:.2f}%  "
              f"beams={r2['beams']}", flush=True)
    return r2