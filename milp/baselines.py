#!/usr/bin/env python3
"""
baselines.py — greedy and GA baselines on the SAME Instance the MILP uses,
so all methods face identical candidates, eligibility, SE, and constraints.

Greedy : classic lazy greedy for capacitated facility location. Repeatedly
         open the candidate with the best NET BENEFIT
             lam * (newly servable demand) - c_j
         given the demand already covered and the per-sector budgets already
         spent, until no candidate has positive net benefit. Then NTN overflow
         per hex up to the beam ceiling; the rest is outage. This mirrors the
         requester's simulator pipeline logic at planning granularity and is
         the natural "density-driven build-out" baseline.
GA     : bitstring over y with feasibility repair (conflicts) and the same
         association evaluator; standard tournament GA — the literature
         baseline, nothing fancy.

Both return the same result dict shape as solve_hex() for the harness.

FIXES IN THIS REVISION
----------------------
* solve_greedy's `served_mask` was written but never actually set (the loop
  did `served_mask[:] = False` then `break`), so the marginal-gain proxy was
  recomputed against the FULL demand every round -- the greedy was ranking
  candidates by raw reachable demand, never by MARGINAL gain. It also called
  the full evaluator 12x per opened site, which on a 45k-demand-point hex is
  hours of wall time for a baseline. Replaced with a correct lazy greedy that
  maintains residual demand and per-sector budgets incrementally.
* lam / c_beam defaulted to 100 / 5 -- placeholders. Now imported from
  hex_milp so every method in the comparison table optimises the SAME
  objective. A baseline scored against a different cost model is not a
  baseline.
* Association is now one shared, tested function (_associate) used by both
  the greedy's final accounting and the GA's fitness.
"""
from __future__ import annotations
import heapq
import time
import numpy as np

from candidate_generator import Instance
from hex_milp import LAM_DEFAULT, C_BEAM_DEFAULT


# ---------------------------------------------------------------------------
def _sector_budgets(inst, y):
    """Per (open site, sector) budget in MHz (== Mbps per bps/Hz)."""
    W = {}
    for j in np.where(np.asarray(y, dtype=bool))[0]:
        w = inst.tiers[inst.cand_tier[j]].bw_hz / 1e6
        for k in range(3):
            W[(int(j), k)] = w
    return W


def _associate(inst, y):
    """Capacity-aware association: demand points in descending demand, each
    takes its best-SE OPEN site that still has room in the serving sector.
    Returns (served_tn array, remaining per-sector budgets)."""
    y = np.asarray(y, dtype=bool)
    U = len(inst.dem_mbps)
    W = _sector_budgets(inst, y)
    served = np.zeros(U)
    for u in np.argsort(-inst.dem_mbps):
        d = float(inst.dem_mbps[u])
        if d <= 0:
            continue
        for j, se, k in sorted(zip(inst.elig_j[u], inst.elig_se[u],
                                   inst.elig_sec[u]), key=lambda t: -t[1]):
            if not y[j]:
                continue
            need = d / max(se, 1e-6)
            if W.get((int(j), k), 0.0) >= need:
                W[(int(j), k)] -= need
                served[u] = d
                break
    return served, W


def _ntn_fill(inst, rest):
    """NTN overflow per res-5 hex, capped at that hex's beam ceiling."""
    U = len(inst.dem_mbps)
    ntn = np.zeros(U)
    by_hex = {}
    for u in range(U):
        if rest[u] > 1e-9:
            by_hex.setdefault(inst.dem_hex[u], []).append(u)
    cap_of = getattr(inst, "beam_cap_of", None) or {}
    for hx, us in by_hex.items():
        cap = float(cap_of.get(hx, inst.beam_cap_mbps))
        for u in sorted(us, key=lambda u: -rest[u]):
            if cap <= 1e-9:
                break
            take = min(rest[u], cap)
            ntn[u] = take
            cap -= take
    beams = sum(1 for hx, us in by_hex.items()
                if any(ntn[u] > 1e-9 for u in us))
    return ntn, beams


def evaluate_placement(inst: Instance, y: np.ndarray,
                       lam: float = LAM_DEFAULT,
                       c_beam: float = C_BEAM_DEFAULT):
    """Objective of a placement under the SAME cost model as solve_hex()."""
    y = np.asarray(y, dtype=bool)
    served_tn, _ = _associate(inst, y)
    rest = inst.dem_mbps - served_tn
    ntn, beams = _ntn_fill(inst, rest)
    out = inst.dem_mbps - served_tn - ntn
    cost = float(inst.cand_cost[y].sum()) + c_beam * beams \
        + lam * float(out.sum())
    return cost, float(served_tn.sum()), float(ntn.sum()), float(out.sum()), beams


def _pack_result(inst, y, lam, c_beam, wall, name):
    y = np.asarray(y, dtype=bool)
    cost, tn, ntn, out, beams = evaluate_placement(inst, y, lam, c_beam)
    total = float(inst.dem_mbps.sum())
    return {
        "status": name, "gap": float("nan"), "objective": cost,
        "wall_s": wall, "n_vars": 0, "n_x": 0,
        "n_conflicts": len(inst.conflict_pairs),
        "opened": {inst.tiers[t].name: int((y & (inst.cand_tier == t)).sum())
                   for t in set(inst.cand_tier)},
        "beams": beams, "served_tn_mbps": tn, "ntn_mbps": ntn,
        "outage_mbps": out, "total_mbps": total,
        "served_pct": 100.0 * (tn + ntn) / max(total, 1e-9),
        "y": y, "no_solution": False,
    }


# ---------------------------------------------------------------------------
def solve_greedy(inst: Instance, lam: float = LAM_DEFAULT,
                 c_beam: float = C_BEAM_DEFAULT, log: bool = False):
    """Lazy greedy: open the candidate with the best marginal net benefit
    until none is positive. O(J log J) heap pops with incremental state --
    no repeated full re-evaluation."""
    t0 = time.time()
    J = len(inst.cand_tier)
    U = len(inst.dem_mbps)

    # invert eligibility: candidate -> [(u, se, sector)]
    by_cand = {}
    for u in range(U):
        for j, se, k in zip(inst.elig_j[u], inst.elig_se[u], inst.elig_sec[u]):
            by_cand.setdefault(int(j), []).append((u, float(se), int(k)))
    for j in by_cand:
        by_cand[j].sort(key=lambda t: -t[1])      # best SE first

    conflicts = {}
    for a, b in inst.conflict_pairs:
        conflicts.setdefault(int(a), set()).add(int(b))
        conflicts.setdefault(int(b), set()).add(int(a))

    y = np.zeros(J, dtype=bool)
    blocked = np.zeros(J, dtype=bool)
    # candidates the caller pre-declared open (neighbour-owned, cost 0)
    fo = getattr(inst, "fixed_open", None)
    if fo is not None:
        y |= np.asarray(fo, dtype=bool)[:J]
    remaining = inst.dem_mbps.astype(float).copy()
    budgets = {}      # (j,k) -> MHz left

    def _open_budget(j):
        w = inst.tiers[inst.cand_tier[j]].bw_hz / 1e6
        ext = (getattr(inst, "ext_residual", {}) or {}).get(j)
        return [float(ext[k]) if ext else w for k in range(3)]

    def _gain(j):
        """Marginal demand candidate j could newly serve, and the plan."""
        bud = _open_budget(j)
        g, plan = 0.0, []
        for (u, se, k) in by_cand.get(j, ()):
            r = remaining[u]
            if r <= 1e-9 or bud[k] <= 1e-9:
                continue
            take = min(r, se * bud[k])
            if take <= 1e-9:
                continue
            bud[k] -= take / max(se, 1e-6)
            g += take
            plan.append((u, take))
        return g, plan, bud

    # seed the budgets of any pre-opened sites by letting them absorb demand
    for j in np.where(y)[0]:
        g, plan, bud = _gain(int(j))
        for u, take in plan:
            remaining[u] -= take
        for k in range(3):
            budgets[(int(j), k)] = bud[k]

    # lazy greedy heap: net benefit is monotone non-increasing as `remaining`
    # shrinks, so a stale value is a valid UPPER bound (standard CELF).
    heap = []
    for j in range(J):
        if y[j] or j not in by_cand:
            continue
        g, _plan, _bud = _gain(j)
        net = lam * g - float(inst.cand_cost[j])
        if net > 0:
            heapq.heappush(heap, (-net, j))

    n_open = 0
    while heap:
        neg, j = heapq.heappop(heap)
        if y[j] or blocked[j]:
            continue
        g, plan, bud = _gain(j)
        net = lam * g - float(inst.cand_cost[j])
        if net <= 0:
            continue
        # still the best after refresh? (compare against next stale bound)
        if heap and net < -heap[0][0] - 1e-9:
            heapq.heappush(heap, (-net, j))
            continue
        y[j] = True
        n_open += 1
        for u, take in plan:
            remaining[u] -= take
        for k in range(3):
            budgets[(j, k)] = bud[k]
        for k in conflicts.get(j, ()):
            blocked[k] = True
        if log and n_open % 200 == 0:
            print(f"    [greedy] {n_open} sites, residual demand "
                  f"{remaining.sum()/1e3:,.1f} Gbps", flush=True)

    if log:
        print(f"    [greedy] opened {n_open} sites in {time.time()-t0:.1f}s",
              flush=True)
    return _pack_result(inst, y, lam, c_beam, time.time() - t0, "greedy")


# ---------------------------------------------------------------------------
def solve_ga(inst: Instance, lam: float = LAM_DEFAULT,
             c_beam: float = C_BEAM_DEFAULT,
             pop: int = 30, gens: int = 40, seed: int = 0,
             log: bool = False):
    """Tournament GA over y with conflict repair.

    WARNING: fitness is a full association pass, O(U*K). At pop=30, gens=40
    that is 2,400 passes -- fine for a benchmark tile of a few thousand demand
    points, prohibitive on a 45k-point province hex. Use it on the tile
    instances the paper's comparison table reports, not on the full hex.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    J = len(inst.cand_tier)
    conflicts = [tuple(int(v) for v in p) for p in inst.conflict_pairs]

    def repair(y):
        y = np.asarray(y, dtype=bool)
        for a, b in conflicts:
            if y[a] and y[b]:
                y[b if rng.random() < 0.5 else a] = False
        return y

    def fitness(y):
        return evaluate_placement(inst, y, lam, c_beam)[0]

    P = [repair(rng.random(J) < 0.15) for _ in range(pop)]
    F = [fitness(y) for y in P]
    for gen in range(gens):
        newP = []
        for _ in range(pop):
            a, b = rng.integers(0, pop, 2)
            pa = P[a] if F[a] < F[b] else P[b]
            c, d = rng.integers(0, pop, 2)
            pb = P[c] if F[c] < F[d] else P[d]
            cut = int(rng.integers(1, max(J, 2)))
            child = np.concatenate([pa[:cut], pb[cut:]]).copy()
            flip = rng.random(J) < (4.0 / max(J, 1))
            child[flip] = ~child[flip]
            newP.append(repair(child))
        newF = [fitness(y) for y in newP]
        allP = P + newP; allF = F + newF
        idx = np.argsort(allF)[:pop]
        P = [allP[i] for i in idx]; F = [allF[i] for i in idx]
        if log and (gen + 1) % 10 == 0:
            print(f"    [ga] gen {gen+1}: best obj {F[0]:,.0f}", flush=True)
    best = P[int(np.argmin(F))]
    return _pack_result(inst, best, lam, c_beam, time.time() - t0, "ga")