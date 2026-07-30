#!/usr/bin/env python3
"""
hex_milp.py
===========
Per-hex placement MILP in HiGHS (highspy), implementing formulation (8):

  min  sum_j c_j y_j + c_beam sum_h b_h + lambda sum_u o_u d_u
  s.t. sum_{j in N(u)} x_uj + z_u + o_u = 1            (assignment)
       x_uj <= y_j                                      (open sites only)
       sum_u (d_u/eta_uj) x_uj <= W_t(j) y_j            (TN bandwidth)
       sum_{u in U_h} d_u z_u <= C_beam b_h             (NTN one-beam ceiling)
       sum_h b_h <= B_max                               (beam budget)
       y_j + y_k <= 1 for conflict pairs                (min inter-site dist)
       y, b binary;  x, z, o in [0,1]  (demand points aggregate many users ->
                                        fractional service is physical)

Variables x are SPARSE: only (u,j) with j in the pruned eligibility set.
"""
from __future__ import annotations
import time
import numpy as np
import highspy

from candidate_generator import Instance


def solve_hex(inst: Instance,
              lam: float = 1309.0,           # USD per Mbps unserved per YEAR
              c_beam: float = 49813.0,      # USD/yr per persistent LEO beam (Toka 2024 Tab.3)
              B_max: int = 10**9,          # beam budget (per-hex solve: large)
              mip_gap: float = 0.02,
              time_limit_s: float = 600.0,
              threads: int = 0,
              max_outage_frac=None,        # None = penalty only (lambda);
                                           # 0.0 = HARD zero-outage (model is
                                           # infeasible if demand cannot be met
                                           # -> proves genuine saturation)
              require_coverage: bool = False,  # force >=1 open eligible site
                                           # per demand point (set-covering row)
              force_serve=None,            # iterable of demand-point indices
                                           # that MUST be fully served (o_u=0).
                                           # This is the scenario cut done
                                           # properly: the model cannot answer
                                           # a margin increase by dropping the
                                           # very point we asked it to protect.
              demand_scale=None,           # (U,) multiplier on provisioned
                                           # demand; >1 = build margin HERE.
                                           # Driven by measured drops.
              cap_factor=None,             # {cand j: rho_j in (0,1]} measured
                                           # by the SIMULATOR: fraction of the
                                           # MILP-assigned load site j actually
                                           # delivered. Derates (8e).
              outage_budget_mbps=None,     # hard cap on TOTAL outage [Mbps]
                                           # (the eps-constraint, Eq. 8b)
              _zero_site_costs=False,      # stage-1 helper: ignore build cost
              fix_closed_mask=None,        # bool (J,): force y_j = 0
              sector_mult: float = 3.0,    # kept for API compat; capacity is
                                           # now PER-SECTOR (W_t each), not 3W pooled
              log: bool = True):
    J = len(inst.cand_tier)
    U = len(inst.dem_mbps)
    _ds = (np.ones(U) if demand_scale is None
           else np.asarray(demand_scale, dtype=float).reshape(-1)[:U])
    H = len(inst.hex_ids)
    hex_index = {h: i for i, h in enumerate(inst.hex_ids)}

    # ---- variable layout: [ y_0..y_{J-1} | b_0..b_{H-1} | z_0.. | o_0.. | x... ]
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
        h.setOptionValue("threads", threads)
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

    if fix_closed_mask is not None:
        for j in np.where(np.asarray(fix_closed_mask))[0]:
            ub[iy(int(j))] = 0.0
    fo = getattr(inst, "fixed_open", None)
    if fo is not None:
        for j in np.where(np.asarray(fo))[0]:
            lb[iy(int(j))] = 1.0            # neighbor-owned: forced open
    h.addVars(N, lb, ub)
    h.changeColsCost(N, np.arange(N, dtype=np.int32), cost)
    h.changeColsIntegrality(N, np.arange(N, dtype=np.int32), integrality)

    # ---- constraints --------------------------------------------------------
    # (8c) assignment: sum_j x_uj + z_u + o_u = 1
    for u in range(U):
        idx = [x_index[(u, j)] for j in inst.elig_j[u]] + [iz(u), io(u)]
        val = [1.0] * len(idx)
        h.addRow(1.0, 1.0, len(idx),
                 np.array(idx, dtype=np.int32), np.array(val))

    # (8d) x_uj <= y_j
    for (u, j), col in x_index.items():
        h.addRow(-highspy.kHighsInf, 0.0, 2,
                 np.array([col, iy(j)], dtype=np.int32),
                 np.array([1.0, -1.0]))

    # (8e) TN bandwidth PER SECTOR: users are wedge-assigned by geometry
    # (3GPP 30/150/270 boresights); each sector has its own W_t budget.
    # This replaces the pooled 3W relaxation, which over-credits sites with
    # angularly imbalanced demand (e.g. at dense-zone boundaries).
    per_sector = {}
    for u in range(U):
        for j, se, k in zip(inst.elig_j[u], inst.elig_se[u], inst.elig_sec[u]):
            per_sector.setdefault((j, k), []).append(
                (x_index[(u, j)],
                 float(inst.dem_mbps[u]) * _ds[u] / max(se, 1e-6)))
    ext_res = getattr(inst, "ext_residual", {}) or {}
    for (j, k), terms in per_sector.items():
        if j in ext_res:
            W_mbps_hz = float(ext_res[j][k])     # neighbor's RESIDUAL only
        else:
            W_mbps_hz = inst.tiers[inst.cand_tier[j]].bw_hz / 1e6  # ONE sector
        if cap_factor is not None and j in cap_factor:
            W_mbps_hz *= float(cap_factor[j])    # simulator-measured derating
        idx = np.array([c for c, _ in terms] + [iy(j)], dtype=np.int32)
        val = np.array([v for _, v in terms] + [-W_mbps_hz])
        h.addRow(-highspy.kHighsInf, 0.0, len(idx), idx, val)

    # (8f) NTN ceiling per hex: sum_{u in U_h} d_u z_u - C_beam b_h <= 0
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
    # has any eligible candidate. Structurally eliminates "no tower in range"
    # drops; infeasible only if a demand point has no candidate at all.
    if require_coverage:
        for u in range(U):
            if not inst.elig_j[u]:
                continue
            idx = np.array(sorted({iy(j) for j in inst.elig_j[u]}),
                           dtype=np.int32)
            val = np.ones(len(idx))
            h.addRow(1.0, highspy.kHighsInf, len(idx), idx, val)

    # FORCE-SERVE: o_u = 0 for protected demand points. Infeasibility here is
    # the honest answer ("this point cannot be served on this candidate set"),
    # not something to be silently traded away against the outage penalty.
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

    # ---- solve ---------------------------------------------------------------
    t0 = time.time()
    h.run()
    wall = time.time() - t0
    info = h.getInfo()
    sol = h.getSolution()
    v = np.array(sol.col_value)

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
        "status": h.modelStatusToString(h.getModelStatus()),
        "gap": getattr(info, "mip_gap", float("nan")),
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
    }


def sector_usage_mhz(inst, res):
    """Per opened site: spectrum consumed per sector [MHz]*3, from the x
    solution. Used by the province driver to hand neighbors the RESIDUAL."""
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
      no_eligible  : demand point has an EMPTY eligibility set (no candidate in
                     range at all) -> coverage/candidate-generation limited
      elig_all_open_full : every eligible candidate IS open but out of sector
                     bandwidth -> genuine capacity saturation
      elig_not_opened    : some eligible candidate was NOT opened -> the model
                     chose outage over building (lambda too low / conflicts
                     blocked it) -> a MODEL knob, not physics
    """
    v, xi = res["x_val"], res["x_index"]
    y = res["y"]
    out = {"no_eligible": 0.0, "elig_all_open_full": 0.0,
           "elig_not_opened": 0.0, "n_pts": 0}
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
        else:
            out["elig_not_opened"] += d
    return out


def solve_hex_min_outage(inst, lam=100.0, c_beam=5.0, mip_gap=0.02,
                         time_limit_s=600.0, threads=0, slack=1.02,
                         log=True, **kw):
    """LEXICOGRAPHIC solve — the honest way to ask for zero outage.

    Stage 1: minimise TOTAL OUTAGE alone (costs zeroed) -> D_min, the provably
             smallest unserved demand any legal deployment can achieve.
             D_min == 0  => zero outage IS attainable.
             D_min  > 0  => zero outage is IMPOSSIBLE (certificate), and the
             value quantifies the saturation.
    Stage 2: re-solve minimising deployment cost subject to
             total outage <= slack * D_min, so you also get the CHEAPEST
             deployment achieving (essentially) that best outage.

    Unlike a hard o_u = 0 bound, this never returns a bare 'Infeasible': you
    always get a placement plus the exact best-attainable outage.
    """
    # ---- stage 1: pure outage minimisation (cost/beam weights -> 0) --------
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

    # ---- stage 2: cheapest deployment holding that outage level ------------
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