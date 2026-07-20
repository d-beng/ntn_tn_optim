#!/usr/bin/env python3
"""
run_tile.py — Day-1 end-to-end test on a synthetic 10K-user tile.
Synthetic geography: one dense urban core + suburbs + rural scatter inside
one H3 res-5 hex near Toronto. Verifies: candidate generation, conflict
pairs, eligibility, MILP solve, and the served/cost/beam report.
On the cluster, replace make_synthetic_users() with a WorldPop hex extract.
"""
import numpy as np
import h3

from candidate_generator import build_instance, DEFAULT_TIERS
from hex_milp import solve_hex


def make_synthetic_users(n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    hex_id = h3.latlng_to_cell(43.70, -79.40, 5)
    clat, clon = h3.cell_to_latlng(hex_id)
    lat, lon = [], []
    # 60% dense core (sigma ~1.2 km), 30% suburb (4 km), 10% rural (8 km)
    for frac, sig_km in [(0.6, 1.2), (0.3, 4.0), (0.1, 8.0)]:
        k = int(n * frac)
        lat.append(clat + rng.normal(0, sig_km / 111.0, k))
        lon.append(clon + rng.normal(0, sig_km / (111.0 * np.cos(np.radians(clat))), k))
    lat = np.concatenate(lat); lon = np.concatenate(lon)
    mbps = np.full(n, 0.385)          # ITU busy-hour dimensioning value
    return hex_id, lat, lon, mbps


def main():
    hex_id, lat, lon, mbps = make_synthetic_users()
    print(f"hex {hex_id}: {len(lat):,} users, demand {mbps.sum():,.0f} Mbps")

    inst = build_instance(lat, lon, mbps, hex_id,
                          rho_cand=1.0, rho_dep=0.95,   # COARSE pass; refine locally later
                          agg_res=9, K_elig=6)
    print(f"candidates: {len(inst.cand_tier):,} "
          f"({ {inst.tiers[t].name: int((inst.cand_tier==t).sum()) for t in set(inst.cand_tier)} })")
    print(f"demand points: {len(inst.dem_mbps):,}  "
          f"conflict pairs: {len(inst.conflict_pairs):,}  "
          f"beam cap: {inst.beam_cap_mbps:.0f} Mbps")

    res = solve_hex(inst, lam=100.0, c_beam=5.0,
                    mip_gap=0.02, time_limit_s=600, log=False)

    print("\n=== MILP RESULT ===")
    for k in ["status", "gap", "objective", "wall_s", "n_vars", "n_x"]:
        print(f"  {k:12s}: {res[k]}")
    print(f"  opened      : {res['opened']}")
    print(f"  beams       : {res['beams']}")
    print(f"  served TN   : {res['served_tn_mbps']:,.0f} Mbps")
    print(f"  served NTN  : {res['ntn_mbps']:,.0f} Mbps")
    print(f"  outage      : {res['outage_mbps']:,.0f} Mbps")
    print(f"  served %    : {res['served_pct']:.2f}%")


if __name__ == "__main__":
    main()
