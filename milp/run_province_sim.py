#!/usr/bin/env python3
"""
run_province_sim.py — validate a PROVINCE-SCALE placement with the full 3GPP
simulator.

The pipeline is three stages and this is the third:

  1. province_solver.py   per-hex MILP (+ halos, + Jacobi border sweep)
                          -> province_placement.csv   (hex, lat, lon, tier)
  2. milp_to_bs_csv.py    -> milp_bs.csv  (3 sector cells per site, full RF)
  3. THIS SCRIPT          load milp_bs.csv, run the simulator on ALL users

Why a separate driver: smoke_real_sim.py builds BaseStation objects from an
in-memory Instance for ONE hex. At province scale the placement comes from a
stitched CSV covering thousands of hexes and there is no single Instance, so
the towers are rebuilt from the CSV instead.

MOBILITY DETERMINISM
--------------------
User.move() is seeded per (seed, user_id, hour). The MILP in stage 1 and the
simulator here therefore see IDENTICAL positions, which is what makes the
comparison meaningful -- at hour 20 the day branch applies and ~40% of users
relocate per unseeded call, so without this the plan is graded on a different
realisation than it was built for.

USAGE
    export PYTHONPATH=/Utilisateurs/dbenguer/ntn_tn_optim/src
    python run_province_sim.py workers=200 +bs_csv=milp_bs.csv
"""
from __future__ import annotations
import json
import math
import pickle
import sys
import time
from collections import Counter

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict

from hybrid_ntn_optimizer.core.types import OrbitType, WalkerParameters
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region

sys.path.insert(0, ".")

DEF_USERS = "/Utilisateurs/dbenguer/ntn_tn_optim/data/users.pkl"
ENUM_OF = {"RMA": "RMA", "UMA": "UMA", "UMI": "UMI", "UMI_MMW": "UMI"}


def build_base_stations_from_csv(path, h3_resolution: int = 5, log: bool = True):
    """Rebuild BaseStation objects from a sector-level CSV.

    Accepts either milp_bs.csv (from milp_to_bs_csv.py) or the
    base_stations.csv the simulator writes -- they share a schema. Every RF
    column is read from the file, so the simulator judges the placement under
    exactly the physics the MILP used.
    """
    from hybrid_ntn_optimizer.models.base_station import (BaseStation,
                                                          DeploymentScenario)
    df = pd.read_csv(path)
    need = ["lat", "lon", "scenario", "coverage_radius_km", "p_tx_dbm",
            "g_tx_dbi", "carrier_freq_hz", "bandwidth_hz", "bs_height_m",
            "shadow_sigma_los_db", "shadow_sigma_nlos_db",
            "interference_cutoff_m", "min_user_dist_m"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}. Generate it with "
                         f"milp_to_bs_csv.py, which writes the full RF set.")

    bss = []
    for i, r in df.iterrows():
        tier = str(r["scenario"])
        bs = BaseStation(
            bs_id=int(r.get("bs_id", i)),
            lat=float(r["lat"]), lon=float(r["lon"]),
            scenario=DeploymentScenario[ENUM_OF.get(tier, tier)],
            p_tx_dbm=float(r["p_tx_dbm"]), g_tx_dbi=float(r["g_tx_dbi"]),
            carrier_freq_hz=float(r["carrier_freq_hz"]),
            total_bandwidth_hz=float(r["bandwidth_hz"]),
            capacity_mbps=0.0,
            bs_height_m=float(r["bs_height_m"]),
            shadow_sigma_los_db=float(r["shadow_sigma_los_db"]),
            shadow_sigma_nlos_db=float(r["shadow_sigma_nlos_db"]),
            interference_cutoff_m=float(r["interference_cutoff_m"]),
            coverage_radius_km=float(r["coverage_radius_km"]),
            min_user_dist_m=float(r["min_user_dist_m"]),
            use_physical_radius=True)
        az = r.get("sector_azimuth_deg", None)
        bs.site_id = int(r.get("site_id", i))
        bs.sector_azimuth_deg = (None if az is None or (isinstance(az, float)
                                                        and math.isnan(az))
                                 else float(az))
        bs.num_sectors = int(r.get("num_sectors", 3))
        bs.assigned_user_count = 0
        bs.cluster_density = 0.0
        bs.raw_cluster_radius_km = float(r["coverage_radius_km"])
        bs.discovery_radius_km = float(r["coverage_radius_km"])
        bs.discovery_cluster_size = 0
        bs.area_class = "TN-Service-Area"
        bs.set_resolution(h3_resolution)
        bss.append(bs)

    if log:
        mix = Counter(str(b.scenario.name) for b in bss)
        nsite = len({int(b.site_id) for b in bss})
        print(f"    {len(bss):,} sector cells / {nsite:,} sites from {path}")
        print(f"    tier mix (cells): {dict(mix)}", flush=True)
    return bss


@hydra.main(version_base=None,
            config_path="/Utilisateurs/dbenguer/ntn_tn_optim/configs",
            config_name="base")
def main(cfg: DictConfig):
    from hybrid_ntn_optimizer.models.scenario import Region
    from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
    from full_pipeline_hooks import run_single_hour, user_result_fields

    bs_csv = str(cfg.get("bs_csv", "milp_bs.csv"))
    users_path = str(cfg.get("users", DEF_USERS))
    hour = float(cfg.get("hour", 20.0))
    workers = int(cfg.get("workers", 8))

    # ---------- region + constellation ----------
    print("[A] building region / constellation ...", flush=True)
    with open(to_absolute_path(cfg.scenario.geojson_path)) as f:
        geometry = json.load(f)
    region = Region(name=cfg.scenario.name, geojson_geometry=geometry,
                    h3_resolution=cfg.scenario.h3_resolution)
    tessellate_region(region, pad_edges=True)
    leos, total_sats = [], 0
    for _k, s in cfg.constellation.shells.items():
        wp = WalkerParameters(total_satellites=s.total_satellites,
                              num_planes=s.num_planes, phasing=s.phasing,
                              inclination_deg=s.inclination_deg,
                              altitude_km=s.altitude_km,
                              orbit_type=OrbitType.LEO)
        leos.append(LEOConstellation(
            params=wp, name=s.name,
            eirp_dbw=cfg.constellation.get("eirp_dbw", 40.0),
            g_t_db=cfg.constellation.get("g_t_db", 10.0),
            max_spot_beams=cfg.constellation.get("max_spot_beams", 15),
            beam_radius_nadir_km=cfg.constellation.get("beam_radius_nadir_km", 200.0),
            max_steering_angle_deg=cfg.constellation.get("max_steering_angle_deg", 45.0)))
        total_sats += leos[-1].num_satellites
    print(f"    {total_sats} sats in {len(leos)} shells", flush=True)

    # ---------- towers ----------
    print(f"[B] towers from {bs_csv} ...", flush=True)
    bss = build_base_stations_from_csv(bs_csv, cfg.scenario.h3_resolution)

    # ---------- ALL users ----------
    print(f"[.] loading {users_path} ...", flush=True)
    t0 = time.time()
    with open(users_path, "rb") as f:
        users = pickle.load(f)
    print(f"    {len(users):,} users in {time.time()-t0:.0f}s", flush=True)
    print(f"[.] moving users to hour {hour} ...", flush=True)
    for u in users:
        u.move(hour, 5)
    dem_tot = sum(u.get_demand_at_time(hour) for u in users)
    print(f"    province demand {dem_tot/1e3:,.1f} Gbps "
          f"({dem_tot/len(users):.3f} Mbps/user)", flush=True)

    # ---------- one simulated hour over the whole province ----------
    import copy
    sim_cfg = copy.deepcopy(cfg)
    try:
        with open_dict(sim_cfg):
            sim_cfg.simulation.duration_s = 72000
            sim_cfg.simulation.time_step_s = 3600
            sim_cfg.simulation.num_workers = workers
    except Exception as e:
        print(f"    (could not force single-hour cfg: {e})", flush=True)
    t0 = time.time()
    run_single_hour(sim_cfg, users, bss, leos, region)
    print(f"[C] province simulation done in {time.time()-t0:.0f}s", flush=True)

    # ---------- census ----------
    reasons, short_by = Counter(), Counter()
    drop = served = offered = 0.0
    for u in users:
        _b, _se, got, d, _dr = user_result_fields(u)
        offered += d
        served += got
        if got + 1e-9 >= d:
            continue
        r = str(getattr(u, "tn_reason", "?") or "?")
        reasons[r] += 1
        short_by[r] += (d - got)
        drop += (d - got)
    print("\n--- WHY USERS WERE SHORT (province) ---", flush=True)
    for r, n in reasons.most_common(10):
        print(f"   {n:>10,} users  {short_by[r]/1e3:>9.1f} Gbps  {r}")
    print(f"\nPROVINCE TOTAL: {drop/1e3:,.1f} Gbps dropped of "
          f"{offered/1e3:,.1f} Gbps offered ({100*drop/max(offered,1e-9):.3f}%)",
          flush=True)

    # ---------- drop map ----------
    try:
        from plot_drops import export_and_plot_drops
        export_and_plot_drops(users, bss, prefix="drops_province",
                              title=f"ONTARIO  |  {len({int(b.site_id) for b in bss}):,}"
                                    f" sites  |  {100*drop/max(offered,1e-9):.3f}% dropped")
    except Exception as e:
        print(f"(drop plot skipped: {e})", flush=True)

    print("\nPROVINCE SIMULATION COMPLETE")


if __name__ == "__main__":
    main()