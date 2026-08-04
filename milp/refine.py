#!/usr/bin/env python3
"""
refine.py — coarse-to-fine candidate refinement.

After the coarse solve (rho_cand=1.0), inject FINE candidates (spacing
rho_fine, down to ~data resolution) ONLY inside a ring around each opened
site, add clique conflicts "at most one per fine neighbourhood", and re-solve
with everything else fixed closed. This reaches pixel-level placement
precision exactly where it matters, without the global fine-lattice symmetry
blowup (measured: rho=0.3 everywhere -> 99.7% gap; coarse -> optimal in 2 s).

TWO BUGS FIXED IN THIS REVISION
-------------------------------
1. refine_instance() built its return Instance WITHOUT beam_cap_of,
   fixed_open or ext_residual. Those are non-default dataclass fields, so the
   call raised TypeError -- refinement could never run. All three are now
   carried through, with fixed_open extended (fine children are never
   forced-open) and ext_residual index-stable (fine candidates are appended
   AFTER the external ones, so existing keys stay valid).
2. Fine-candidate eligibility used the RAW tier radius while build_instance
   uses the aggregation-corrected EFFECTIVE radius (tier radius minus the
   demand-hex circumradius x agg_safety). The fine pass was therefore
   crediting refined candidates with coverage the coarse pass had correctly
   refused them. It now reads inst.r_eff_km.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree

from candidate_generator import Instance, tri_lattice, sector_of
from hex_milp import solve_hex, LAM_DEFAULT, C_BEAM_DEFAULT


def refine_instance(inst: Instance, y_coarse: np.ndarray,
                    user_xy: np.ndarray,
                    rho_fine: float = 0.30,
                    se_fn=None) -> Instance:
    """Return a NEW Instance = coarse candidates + fine candidates around the
    opened coarse sites. Fine candidates of a neighbourhood conflict with each
    other AND with their parent coarse candidate (pick the best position)."""
    assert se_fn is not None, "pass the same se_fn used for the coarse pass"
    tiers = inst.tiers
    y_coarse = np.asarray(y_coarse, dtype=bool)
    add_xy, add_tier, add_cost, add_parent = [], [], [], []

    for j in np.where(y_coarse)[0]:
        t = tiers[inst.cand_tier[j]]
        spacing_fine = t.radius_km * math.sqrt(3.0) * rho_fine
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
            add_parent.append(int(j))

    if not add_xy:
        return inst

    J0 = len(inst.cand_tier)
    cand_xy = np.vstack([inst.cand_xy, np.array(add_xy)])
    cand_tier = np.concatenate([inst.cand_tier,
                                np.array(add_tier, dtype=np.int32)])
    cand_cost = np.concatenate([inst.cand_cost, np.array(add_cost)])

    # conflicts: original pairs + "one per neighbourhood" cliques
    pairs = [tuple(int(v) for v in p) for p in inst.conflict_pairs]
    fam = {}
    for off, parent in enumerate(add_parent):
        fam.setdefault(parent, [parent]).append(J0 + off)
    for members in fam.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pairs.append((members[a], members[b]))
    conflict_pairs = np.array(sorted(set(pairs)), dtype=np.int64)

    # extend eligibility with the new candidates, at the SAME effective radius
    # build_instance used (aggregation-corrected, not the raw tier radius).
    r_eff = (list(inst.r_eff_km) if getattr(inst, "r_eff_km", ())
             else [t.radius_km for t in tiers])
    tree = cKDTree(cand_xy[J0:])
    elig_j = [list(e) for e in inst.elig_j]
    elig_se = [list(e) for e in inst.elig_se]
    elig_sec = [list(e) for e in inst.elig_sec]

    rmax = max(r_eff)
    K = max((len(e) for e in inst.elig_j), default=6) or 6
    for u in range(len(inst.dem_mbps)):
        near = tree.query_ball_point(inst.dem_xy[u], rmax)
        for loc in near:
            j = J0 + loc
            t = tiers[cand_tier[j]]
            d = float(np.hypot(*(cand_xy[j] - inst.dem_xy[u])))
            if d > r_eff[cand_tier[j]]:
                continue
            se = se_fn(d, t)
            if se >= 0.5:
                elig_j[u].append(j)
                elig_se[u].append(se)
                elig_sec[u].append(sector_of(cand_xy[j], inst.dem_xy[u]))
        # PRUNE ONLY THE NEW ENTRIES: keep every coarse entry (so the coarse
        # solution stays feasible => the fine pass can only improve) and add
        # at most K best-SE fine children on top.
        n0 = len(inst.elig_j[u])
        if len(elig_j[u]) > n0 + K:
            new_trip = sorted(zip(elig_se[u][n0:], elig_j[u][n0:],
                                  elig_sec[u][n0:]), reverse=True)[:K]
            elig_se[u] = elig_se[u][:n0] + [a for a, _b, _c in new_trip]
            elig_j[u] = elig_j[u][:n0] + [b for _a, b, _c in new_trip]
            elig_sec[u] = elig_sec[u][:n0] + [c for _a, _b, c in new_trip]

    owner = list(inst.cand_owner_hex) + [inst.cand_owner_hex[p]
                                         for p in add_parent]
    # fine children are ordinary payable candidates: never forced open
    fixed_open = np.concatenate([np.asarray(inst.fixed_open, dtype=bool),
                                 np.zeros(len(add_parent), dtype=bool)])
    return Instance(cand_xy=cand_xy, cand_tier=cand_tier, cand_cost=cand_cost,
                    cand_owner_hex=owner, tiers=tiers, lat0=inst.lat0,
                    dem_xy=inst.dem_xy, dem_mbps=inst.dem_mbps,
                    dem_hex=inst.dem_hex,
                    elig_j=elig_j, elig_se=elig_se, elig_sec=elig_sec,
                    conflict_pairs=conflict_pairs,
                    beam_cap_of=dict(inst.beam_cap_of),
                    fixed_open=fixed_open,
                    ext_residual=dict(inst.ext_residual),
                    hex_ids=list(inst.hex_ids),
                    beam_cap_mbps=inst.beam_cap_mbps,
                    agg_res=inst.agg_res, agg_safety=inst.agg_safety,
                    r_eff_km=inst.r_eff_km)


def solve_with_refinement(inst, user_xy, se_fn, rho_fine=0.30,
                          lam=LAM_DEFAULT, c_beam=C_BEAM_DEFAULT,
                          mip_gap=0.02, time_limit_s=600.0, threads=0,
                          log=True):
    res_c = solve_hex(inst, lam=lam, c_beam=c_beam, mip_gap=mip_gap,
                      time_limit_s=time_limit_s, threads=threads, log=False)
    if log:
        print(f"  coarse: obj={res_c['objective']:.0f} "
              f"served={res_c['served_pct']:.2f}% "
              f"opened={res_c['opened']} ({res_c['wall_s']:.0f}s)")
    if res_c.get("no_solution") or not res_c["y"].any():
        if log:
            print("  coarse solve produced no placement -- skipping refinement")
        return res_c, res_c, inst

    inst_f = refine_instance(inst, res_c["y"], user_xy,
                             rho_fine=rho_fine, se_fn=se_fn)
    if inst_f is inst:
        if log:
            print("  no fine candidates generated -- returning coarse result")
        return res_c, res_c, inst

    # LOCAL REPOSITIONING: only the opened coarse sites and their fine
    # children are free; all other candidates are fixed closed. The fine pass
    # then just picks the best position within each neighbourhood (clique: at
    # most one) and re-optimises association -- small and fast.
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
            / max(abs(res_c["objective"]), 1e-9)
        print(f"  fine  : obj={res_f['objective']:.0f} "
              f"served={res_f['served_pct']:.2f}% "
              f"opened={res_f['opened']} ({res_f['wall_s']:.0f}s)  "
              f"[refinement gain {gain:+.2f}%]")
    return res_c, res_f, inst_f