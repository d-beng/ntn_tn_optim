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
              lam: float = 100.0,          # outage penalty per Mbps
              c_beam: float = 5.0,         # beam activation cost
              B_max: int = 10**9,          # beam budget (per-hex solve: large)
              mip_gap: float = 0.02,
              time_limit_s: float = 600.0,
              threads: int = 0,
              fix_closed_mask=None,        # bool (J,): force y_j = 0
              sector_mult: float = 3.0,    # kept for API compat; capacity is
                                           # now PER-SECTOR (W_t each), not 3W pooled
              log: bool = True):
    J = len(inst.cand_tier)
    U = len(inst.dem_mbps)
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
        cost[iy(j)] = float(inst.cand_cost[j])
        integrality[iy(j)] = 1
    for hh in range(H):
        cost[ib(hh)] = c_beam
        integrality[ib(hh)] = 1
    for u in range(U):
        cost[io(u)] = lam * float(inst.dem_mbps[u])

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
                (x_index[(u, j)], float(inst.dem_mbps[u]) / max(se, 1e-6)))
    ext_res = getattr(inst, "ext_residual", {}) or {}
    for (j, k), terms in per_sector.items():
        if j in ext_res:
            W_mbps_hz = float(ext_res[j][k])     # neighbor's RESIDUAL only
        else:
            W_mbps_hz = inst.tiers[inst.cand_tier[j]].bw_hz / 1e6  # ONE sector
        idx = np.array([c for c, _ in terms] + [iy(j)], dtype=np.int32)
        val = np.array([v for _, v in terms] + [-W_mbps_hz])
        h.addRow(-highspy.kHighsInf, 0.0, len(idx), idx, val)

    # (8f) NTN ceiling per hex: sum_{u in U_h} d_u z_u - C_beam b_h <= 0
    users_of_hex = {}
    for u, hx in enumerate(inst.dem_hex):
        users_of_hex.setdefault(hx, []).append(u)
    for hx, us in users_of_hex.items():
        hh = hex_index[hx]
        idx = np.array([iz(u) for u in us] + [ib(hh)], dtype=np.int32)
        val = np.array([float(inst.dem_mbps[u]) for u in us]
                       + [-inst.beam_cap_mbps])
        h.addRow(-highspy.kHighsInf, 0.0, len(idx), idx, val)

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

    return {
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
