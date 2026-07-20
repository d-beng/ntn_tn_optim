#!/usr/bin/env python3
"""
run_real_tile.py — solve the placement MILP on ONE REAL hex from your
existing codebase (users.pkl + real TR 38.901 link budget from sinr.py).

Usage (on the cluster, inside ntn_env, with highspy installed):
    pip install highspy
    python run_real_tile.py                          # densest hex, auto
    python run_real_tile.py --hex 85062e6ffffffff    # specific H3 res-5 hex
    python run_real_tile.py --users /path/users.pkl --config /path/5g_base.yaml

Outputs:
    tile report on stdout  +  milp_placement_<hex>.csv
    (lat, lon, tier of every opened site — feed this to the simulator
     for the validation step)

Place this file next to candidate_generator.py and hex_milp.py, with
PYTHONPATH including your package src ("/home/db3n/Documents/Ph.D./Courses/ntn_tn_test/ntn_tn_optim-master_2/src").
"""
from __future__ import annotations
import argparse, math, pickle, sys, time
from collections import Counter

import numpy as np
import h3
import yaml

from candidate_generator import build_instance, Tier
from hex_milp import solve_hex


# ---------------------------------------------------------------------------
# 1. Tiers from YOUR 5g_base.yaml (single source of truth — no copies)
# ---------------------------------------------------------------------------
TIER_COSTS   = {"RMA": 2.0, "UMA": 2.5, "UMI": 1.0, "UMI_MMW": 1.2}
# density gates: your res-9 rescaled thresholds (override via CLI)
TIER_DENSMIN = {"RMA": 0.0, "UMA": 1600.0, "UMI": 4000.0, "UMI_MMW": 15000.0}

def load_tiers(cfg_path: str, dens_uma: float, dens_umi: float,
               dens_mmw: float) -> tuple:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    scen = cfg["scenarios"]
    TIER_DENSMIN.update({"UMA": dens_uma, "UMI": dens_umi, "UMI_MMW": dens_mmw})
    tiers = []
    for name in ["RMA", "UMA", "UMI", "UMI_MMW"]:
        if name not in scen:
            print(f"  (tier {name} not in config -> skipped)")
            continue
        s = scen[name]
        tiers.append(Tier(
            name=name,
            freq_hz=float(s["carrier_freq_hz"]),
            bw_hz=float(s["bandwidth_hz"]),
            radius_km=float(s["coverage_radius_km"]),
            p_tx_dbm=float(s["p_tx_dbm"]),
            g_tx_dbi=float(s["g_tx_dbi"]),
            h_bs_m=float(s["default_h_bs"]),
            cost=TIER_COSTS[name],
            density_min=TIER_DENSMIN[name],
        ))
    return tuple(tiers)


# ---------------------------------------------------------------------------
# 2. REAL link-budget SE via your sinr.py (Assumption 1: nominal interference
#    = 6 co-tier interferers on a ring at the design ISD; shadowing sigmas
#    set to 0 so planning coefficients are deterministic expectations)
# ---------------------------------------------------------------------------
def make_real_se_fn(cfg_path: str, rho_dep: float = 0.95):
    from hybrid_ntn_optimizer.link_budget.sinr import calculate_tn_sinr_capacity
    from hybrid_ntn_optimizer.models.base_station import DeploymentScenario
    with open(cfg_path) as f:
        scen_cfg = yaml.safe_load(f)["scenarios"]

    cache: dict = {}

    def se_fn(dist_km: float, tier: Tier) -> float:
        # 50 m distance bins -> cheap cache, deterministic coefficients
        key = (tier.name, round(dist_km / 0.05))
        if key in cache:
            return cache[key]
        s = scen_cfg[tier.name]
        enum_key = "UMI" if tier.name == "UMI_MMW" else tier.name
        scen = DeploymentScenario[enum_key]
        isd_m = tier.radius_km * math.sqrt(3.0) * rho_dep * 1000.0
        interferers = [(
            isd_m, scen, float(s["carrier_freq_hz"]), float(s["default_h_bs"]),
            0.0, 0.0,                                # sigmas -> deterministic
            float(s["p_tx_dbm"]), float(s["g_tx_dbi"]),
            0.0, 0.0, None,                          # lat/lon/az unused (omni)
        ) for _ in range(6)]
        try:
            _, _, se, _ = calculate_tn_sinr_capacity(
                bs_height_m=float(s["default_h_bs"]),
                dist_to_serving_m=max(dist_km * 1000.0, float(s["min_user_dist_m"])),
                interferers=interferers,
                shadow_sigma_los_db=0.0, shadow_sigma_nlos_db=0.0,
                scenario=scen,
                p_tx_dbm=float(s["p_tx_dbm"]),
                g_tx_dbi=float(s["g_tx_dbi"]),
                g_rx_ue_dbi=0.0,
                serving_beamforming_gain_db=12.0,
                interferer_beamforming_suppression_db=12.0,
                carrier_freq_hz=float(s["carrier_freq_hz"]),
                bandwidth_hz=float(s["bandwidth_hz"]),
                body_loss_db=3.0,
                noise_figure_db=10.0 if float(s["carrier_freq_hz"]) > 24e9 else 7.0,
                implementation_loss_factor=0.65,
                ue_height_m=1.5,
            )
        except Exception:
            se = 0.0
        cache[key] = float(max(se, 0.0))
        return cache[key]

    return se_fn


# ---------------------------------------------------------------------------
# 3. Hex extraction from users.pkl
# ---------------------------------------------------------------------------
def extract_hex(users_path: str, hex_id: str | None, halo_km: float,
                hour: float = 20.0):
    print(f"  loading {users_path} ... (14.5M objects, be patient)", flush=True)
    t0 = time.time()
    with open(users_path, "rb") as f:
        users = pickle.load(f)
    print(f"  loaded {len(users):,} users in {time.time()-t0:.0f}s", flush=True)

    lats = np.array([u.current_lat for u in users])
    lons = np.array([u.current_lon for u in users])

    cells = [h3.latlng_to_cell(float(a), float(b), 5)
             for a, b in zip(lats, lons)]
    if hex_id is None:
        hex_id, n = Counter(cells).most_common(1)[0]
        print(f"  auto-selected densest hex {hex_id} ({n:,} users)")

    # hex + halo: users in the hex, plus users in ring-1 neighbours within
    # (circumradius + halo) of the hex centre
    clat, clon = h3.cell_to_latlng(hex_id)
    edge_km = h3.average_hexagon_edge_length(5, unit="km")
    r_cut = edge_km + halo_km
    neigh = set(h3.grid_disk(hex_id, 1))
    def hav_km(a1, b1, a2, b2):
        p1, p2 = math.radians(a1), math.radians(a2)
        dp, dl = p2 - p1, math.radians(b2 - b1)
        x = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return 2 * 6371.0088 * math.asin(math.sqrt(x))
    sel = [i for i, c in enumerate(cells)
           if c == hex_id or (c in neigh and
                              hav_km(lats[i], lons[i], clat, clon) <= r_cut)]
    sel = np.array(sel)
    demand = np.array([users[i].get_demand_at_time(hour) for i in sel])
    active = demand >= 0.1
    print(f"  hex+halo users: {len(sel):,}  active at h{hour:.0f}: {active.sum():,}"
          f"  demand: {demand[active].sum():,.0f} Mbps")
    return hex_id, lats[sel][active], lons[sel][active], demand[active]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users",  default="/home/db3n/Documents/Ph.D./Courses/ntn_tn_test/ntn_tn_optim-master_2/data/users.pkl")
    ap.add_argument("--config", default="/home/db3n/Documents/Ph.D./Courses/ntn_tn_test/ntn_tn_optim-master_2/configs/terrestrial/5g_base.yaml")
    ap.add_argument("--hex",    default=None, help="H3 res-5 id (default: densest)")
    ap.add_argument("--halo-km",   type=float, default=2.847)
    ap.add_argument("--rho-cand",  type=float, default=1.0)
    ap.add_argument("--rho-dep",   type=float, default=0.95)
    ap.add_argument("--agg-res",   type=int,   default=9)
    ap.add_argument("--k-elig",    type=int,   default=6)
    ap.add_argument("--dens-uma",  type=float, default=1600.0)
    ap.add_argument("--dens-umi",  type=float, default=4000.0)
    ap.add_argument("--dens-mmw",  type=float, default=15000.0)
    ap.add_argument("--lam",       type=float, default=100.0)
    ap.add_argument("--c-beam",    type=float, default=5.0)
    ap.add_argument("--gap",       type=float, default=0.02)
    ap.add_argument("--time-limit",type=float, default=1800.0)
    ap.add_argument("--threads",   type=int,   default=0)
    ap.add_argument("--beam-bw",   type=float, default=300e6)
    ap.add_argument("--beam-se",   type=float, default=1.77)
    args = ap.parse_args()

    print("=== REAL-TILE MILP ===")
    tiers = load_tiers(args.config, args.dens_uma, args.dens_umi, args.dens_mmw)
    print(f"  tiers: {[t.name for t in tiers]}")
    se_fn = make_real_se_fn(args.config, rho_dep=args.rho_dep)

    hex_id, lat, lon, mbps = extract_hex(args.users, args.hex, args.halo_km)

    t0 = time.time()
    inst = build_instance(lat, lon, mbps, hex_id, tiers=tiers, se_fn=se_fn,
                          rho_cand=args.rho_cand, rho_dep=args.rho_dep,
                          agg_res=args.agg_res, K_elig=args.k_elig,
                          beam_bw_hz=args.beam_bw, beam_se=args.beam_se)
    print(f"  instance: {len(inst.cand_tier):,} candidates "
          f"({ {inst.tiers[t].name: int((inst.cand_tier==t).sum()) for t in set(inst.cand_tier)} }), "
          f"{len(inst.dem_mbps):,} demand pts, "
          f"{len(inst.conflict_pairs):,} conflicts  [{time.time()-t0:.1f}s]")

    res = solve_hex(inst, lam=args.lam, c_beam=args.c_beam,
                    mip_gap=args.gap, time_limit_s=args.time_limit,
                    threads=args.threads, log=True)

    print("\n=== RESULT ===")
    for k in ["status", "gap", "objective", "wall_s", "n_vars"]:
        print(f"  {k:12s}: {res[k]}")
    print(f"  opened      : {res['opened']}")
    print(f"  beams       : {res['beams']}")
    print(f"  TN / NTN / out (Mbps): {res['served_tn_mbps']:,.0f} / "
          f"{res['ntn_mbps']:,.0f} / {res['outage_mbps']:,.0f}")
    print(f"  served %    : {res['served_pct']:.2f}%")

    # placement CSV for simulator validation
    import csv
    from candidate_generator import unproject
    y = res["y"]
    plat, plon = unproject(inst.cand_xy[y, 0], inst.cand_xy[y, 1], inst.lat0)
    out = f"milp_placement_{hex_id}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "tier"])
        for a, b, t in zip(plat, plon, inst.cand_tier[y]):
            w.writerow([f"{a:.6f}", f"{b:.6f}", inst.tiers[t].name])
    print(f"  placement -> {out}  ({int(y.sum())} sites)")


if __name__ == "__main__":
    main()
