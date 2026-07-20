#!/usr/bin/env python3
"""
refine.py — coarse-to-fine candidate refinement.

After the coarse solve (rho_cand=1.0), inject FINE candidates (spacing
rho_fine, down to ~data resolution) ONLY inside a ring around each opened
site, add clique conflicts "at most one per fine neighborhood", warm-start
from the coarse solution, and re-solve. This reaches pixel-level placement
precision exactly where it matters, without the global fine-lattice symmetry
blowup (measured: rho=0.3 everywhere -> 99.7% gap; coarse -> optimal in 2 s).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree

from candidate_generator import Instance, tri_lattice
from hex_milp import solve_hex


def refine_instance(inst: Instance, y_coarse: np.ndarray,
                    user_xy: np.ndarray,
                    rho_fine: float = 0.30,
                    se_fn=None) -> Instance:
    """Return a NEW Instance = coarse candidates + fine candidates around the
    opened coarse sites. Fine candidates of a neighborhood conflict with each
    other AND with their parent coarse candidate (pick the best position)."""
    assert se_fn is not None, "pass the same se_fn used for the coarse pass"
    tiers = inst.tiers
    add_xy, add_tier, add_cost, add_parent = [], [], [], []

    for j in np.where(y_coarse)[0]:
        t = tiers[inst.cand_tier[j]]
        spacing_fine = t.radius_km * math.sqrt(3.0) * rho_fine
        # fine lattice over USER positions within 1 coarse-spacing of site j
        r_ring = t.radius_km * math.sqrt(3.0) * 1.0
        d = np.hypot(user_xy[:, 0] - inst.cand_xy[j, 0],
                     user_xy[:, 1] - inst.cand_xy[j, 1])
        m = d <= r_ring
        if m.sum() < 3:
            continue
        nx, ny, cnt = tri_lattice(user_xy[m, 0], user_xy[m, 1], spacing_fine)
        for a, b in zip(nx, ny):
            if np.hypot(a - inst.cand_xy[j, 0], b - inst.cand_xy[j, 1]) < 1e-6:
                continue                       # coarse node itself
            add_xy.append((a, b))
            add_tier.append(inst.cand_tier[j])
            add_cost.append(inst.cand_cost[j])
            add_parent.append(j)

    if not add_xy:
        return inst

    J0 = len(inst.cand_tier)
    cand_xy = np.vstack([inst.cand_xy, np.array(add_xy)])
    cand_tier = np.concatenate([inst.cand_tier,
                                np.array(add_tier, dtype=np.int32)])
    cand_cost = np.concatenate([inst.cand_cost, np.array(add_cost)])

    # conflicts: original pairs + "one per neighborhood" cliques
    pairs = [tuple(p) for p in inst.conflict_pairs]
    fam = {}
    for off, parent in enumerate(add_parent):
        fam.setdefault(parent, [parent]).append(J0 + off)
    for members in fam.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pairs.append((members[a], members[b]))
    conflict_pairs = np.array(sorted(set(pairs)), dtype=np.int64)

    # extend eligibility with the new candidates
    tree = cKDTree(cand_xy[J0:])
    elig_j = [list(e) for e in inst.elig_j]
    elig_se = [list(e) for e in inst.elig_se]
    elig_sec = [list(e) for e in inst.elig_sec]

    def _sector_of(cand_p, dem_p):
        dx = dem_p[0] - cand_p[0]; dy = dem_p[1] - cand_p[1]
        bearing = math.degrees(math.atan2(dx, dy)) % 360.0
        return int(((bearing - 330.0) % 360.0) // 120.0)

    rmax = max(t.radius_km for t in tiers)
    K = max(len(e) for e in inst.elig_j) if inst.elig_j else 6
    for u in range(len(inst.dem_mbps)):
        near = tree.query_ball_point(inst.dem_xy[u], rmax)
        for loc in near:
            j = J0 + loc
            t = tiers[cand_tier[j]]
            d = float(np.hypot(*(cand_xy[j] - inst.dem_xy[u])))
            if d > t.radius_km:
                continue
            se = se_fn(d, t)
            if se >= 0.5:
                elig_j[u].append(j)
                elig_se[u].append(se)
                elig_sec[u].append(_sector_of(cand_xy[j], inst.dem_xy[u]))
        # PRUNE ONLY THE NEW ENTRIES: keep every coarse entry (so the coarse
        # solution stays feasible => fine pass can only improve) and add at
        # most K best-SE fine children on top.
        n0 = len(inst.elig_j[u])
        if len(elig_j[u]) > n0 + K:
            new_trip = sorted(zip(elig_se[u][n0:], elig_j[u][n0:],
                                  elig_sec[u][n0:]), reverse=True)[:K]
            elig_se[u]  = elig_se[u][:n0] + [a for a, _b, _c in new_trip]
            elig_j[u]   = elig_j[u][:n0]  + [b for _a, b, _c in new_trip]
            elig_sec[u] = elig_sec[u][:n0] + [c for _a, _b, c in new_trip]

    owner = list(inst.cand_owner_hex) + [inst.cand_owner_hex[p]
                                         for p in add_parent]
    return Instance(cand_xy=cand_xy, cand_tier=cand_tier, cand_cost=cand_cost,
                    cand_owner_hex=owner, tiers=tiers, lat0=inst.lat0,
                    dem_xy=inst.dem_xy, dem_mbps=inst.dem_mbps,
                    dem_hex=inst.dem_hex,
                    elig_j=elig_j, elig_se=elig_se, elig_sec=elig_sec,
                    conflict_pairs=conflict_pairs,
                    hex_ids=inst.hex_ids, beam_cap_mbps=inst.beam_cap_mbps)


def solve_with_refinement(inst, user_xy, se_fn, rho_fine=0.30,
                          lam=100.0, c_beam=5.0, mip_gap=0.02,
                          time_limit_s=600.0, threads=0, log=True):
    res_c = solve_hex(inst, lam=lam, c_beam=c_beam, mip_gap=mip_gap,
                      time_limit_s=time_limit_s, threads=threads, log=False)
    if log:
        print(f"  coarse: obj={res_c['objective']:.0f} "
              f"served={res_c['served_pct']:.2f}% "
              f"opened={res_c['opened']} ({res_c['wall_s']:.0f}s)")
    inst_f = refine_instance(inst, res_c["y"], user_xy,
                             rho_fine=rho_fine, se_fn=se_fn)
    # LOCAL REPOSITIONING: only the opened coarse sites and their fine
    # children are free; all other candidates are fixed closed. The fine
    # pass then just picks the best position within each neighborhood
    # (clique: at most one) and re-optimizes association -- small and fast.
    J0 = len(inst.cand_tier)
    Jf = len(inst_f.cand_tier)
    free = np.zeros(Jf, dtype=bool)
    free[np.where(res_c["y"])[0]] = True     # opened coarse parents
    free[J0:] = True                          # their fine children
    fix_closed = ~free
    res_f = solve_hex(inst_f, lam=lam, c_beam=c_beam, mip_gap=mip_gap,
                      time_limit_s=time_limit_s, threads=threads, log=False,
                      fix_closed_mask=fix_closed)
    if log:
        gain = 100.0 * (res_c["objective"] - res_f["objective"]) \
            / max(res_c["objective"], 1e-9)
        print(f"  fine  : obj={res_f['objective']:.0f} "
              f"served={res_f['served_pct']:.2f}% "
              f"opened={res_f['opened']} ({res_f['wall_s']:.0f}s)  "
              f"[refinement gain {gain:+.2f}%]")
    return res_c, res_f, inst_f
