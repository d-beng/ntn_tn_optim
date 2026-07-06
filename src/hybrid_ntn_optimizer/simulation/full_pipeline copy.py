import os
import math
from time import sleep
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from omegaconf import DictConfig

from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation, DeploymentScenario
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.coverage.mapper import map_satellites_to_region

from hybrid_ntn_optimizer.core.utils import haversine_distance, _detect_cpus
from hybrid_ntn_optimizer.link_budget.sinr import calculate_tn_sinr_capacity
from hybrid_ntn_optimizer.allocation.beam_allocator import allocate_ntn_beams
from hybrid_ntn_optimizer.link_budget.sector_antenna import (
    sector_gain_db, in_sector,
)

from scipy.spatial import cKDTree


# ======================================================================
# [PARALLEL + SPATIAL] Worker-side state and function for PHASE 1
# ----------------------------------------------------------------------
# SPEED FIX: previously every user scanned the ENTIRE base-station list
# (O(all cells)) twice — once for serving candidates and once for
# interferers. With sectorization tripling the cell count this became the
# dominant cost. We now build ONE KD-tree over cell positions (in a local
# equirectangular km plane) at pool-init time. Each user then queries only:
#   - serving candidates: cells whose center is within the max coverage radius,
#   - interferers: cells within the max interference cutoff.
# This turns per-user work from O(all cells) into O(nearby cells), so a
# 3-sector deployment costs almost the same as omni.
#
# SECTORIZATION (3GPP TR 38.901 macro geometry; TR 36.942 4.2.1 pattern):
#   - a user may only attach to a sector whose 120-deg wedge contains it;
#   - the serving cell's antenna gain includes its horizontal sector offset;
#   - co-located sectors of the serving SITE (same site_id) are not counted
#     as interferers (they point away);
#   - each remaining interferer sector's pattern offset toward the user is
#     applied. SINR is computed only ONCE per user (for the single serving
#     sector that passes the wedge test); a site's other two sectors are
#     rejected by the cheap angle test before any SINR math.
# ======================================================================
_BASE_STATIONS: List[BaseStation] = []
_G_RX_UE_DBI: float = 0.0
_KD = None
_CELL_XY = None
_LAT0 = 0.0
_MAX_COV_R_KM = 0.0
_MAX_INTF_CUTOFF_M = 0.0
_R_EARTH_KM = 6371.0088


def _project_km(lat, lon, lat0):
    x = math.radians(lon) * math.cos(math.radians(lat0)) * _R_EARTH_KM
    y = math.radians(lat) * _R_EARTH_KM
    return x, y


def _init_attachment_worker(base_stations, g_rx_ue_dbi, lat0,
                            cell_xy, max_cov_r_km, max_intf_cutoff_m):
    """Pool initializer: store the static snapshot + spatial index per worker."""
    global _BASE_STATIONS, _G_RX_UE_DBI, _KD, _CELL_XY, _LAT0
    global _MAX_COV_R_KM, _MAX_INTF_CUTOFF_M
    _BASE_STATIONS = base_stations
    _G_RX_UE_DBI = g_rx_ue_dbi
    _LAT0 = lat0
    _CELL_XY = cell_xy
    _MAX_COV_R_KM = max_cov_r_km
    _MAX_INTF_CUTOFF_M = max_intf_cutoff_m
    if cKDTree is not None and cell_xy is not None and len(cell_xy):
        _KD = cKDTree(cell_xy)
    else:
        _KD = None


def _evaluate_attachment(user_pos):
    """Side-effect-free PHASE 1 inner loop with spatial pre-filtering + sectors.

    Input : (user_lat, user_lon)
    Output: (best_bs_id, best_sinr, best_spec_eff, best_diag)
    """
    u_lat, u_lon = user_pos
    best_bs_id = None
    best_sinr = -999.0
    best_spec_eff = 0.0
    best_diag = {"S_dBm": float('nan'), "I_dBm": float('nan'),
                 "N_dBm": float('nan'), "num_interferers": 0,
                 "IoverN_dB": float('nan')}

    # --- SPATIAL PREFILTER: candidate serving cells + candidate interferers ---
    if _KD is not None:
        ux, uy = _project_km(u_lat, u_lon, _LAT0)
        serving_idx = _KD.query_ball_point((ux, uy), _MAX_COV_R_KM)
        if not serving_idx:
            return (None, best_sinr, best_spec_eff, best_diag)
        intf_idx = _KD.query_ball_point((ux, uy), _MAX_INTF_CUTOFF_M / 1000.0)
        intf_cells = [_BASE_STATIONS[i] for i in intf_idx]
    else:
        serving_idx = range(len(_BASE_STATIONS))
        intf_cells = _BASE_STATIONS

    for si in serving_idx:
        bs = _BASE_STATIONS[si]
        d_m = haversine_distance(u_lat, u_lon, bs.lat, bs.lon)
        if (d_m / 1000.0) > bs.coverage_radius_km:
            continue

        # SECTOR ADMISSION: user must lie inside this sector's wedge (omni passes)
        if not in_sector(bs.lat, bs.lon, getattr(bs, "sector_azimuth_deg", None),
                         u_lat, u_lon):
            continue

        d_m = max(d_m, bs.min_user_dist_m)

        # serving sector antenna gain offset toward the user (<=0 dB; 0 for omni)
        serv_sector_gain = sector_gain_db(
            bs.lat, bs.lon, getattr(bs, "sector_azimuth_deg", None), u_lat, u_lon)

        # interferers: nearby cells within their own cutoff, excluding the
        # serving cell AND all co-located sectors of the serving site.
        interferers = []
        serving_site = getattr(bs, "site_id", -2)
        for other in intf_cells:
            if other.bs_id == bs.bs_id:
                continue
            if getattr(other, "site_id", -1) == serving_site:
                continue
            dist = haversine_distance(u_lat, u_lon, other.lat, other.lon)
            if dist <= other.interference_cutoff_m:
                dist = max(dist, other.min_user_dist_m)
                interferers.append((other, dist))

        sinr_db, capacity_mbps, spec_eff, diag = calculate_tn_sinr_capacity(
            dist_to_serving_m=d_m,
            interferers=interferers,
            scenario=bs.scenario,
            p_tx_dbm=bs.p_tx_dbm,
            g_tx_dbi=bs.g_tx_dbi + serv_sector_gain,
            g_rx_ue_dbi=_G_RX_UE_DBI,
            carrier_freq_hz=bs.carrier_freq_hz,
            bandwidth_hz=bs.total_bandwidth_hz,
            bs_height_m=bs.bs_height_m,
            shadow_sigma_los_db=bs.shadow_sigma_los_db,
            shadow_sigma_nlos_db=bs.shadow_sigma_nlos_db,
            ue_lat=u_lat,
            ue_lon=u_lon,
        )

        if sinr_db > best_sinr:
            best_sinr = sinr_db
            best_spec_eff = spec_eff
            best_bs_id = bs.bs_id
            best_diag = diag

    return (best_bs_id, best_sinr, best_spec_eff, best_diag)


def run_daily_mobility_simulation(
    cfg: DictConfig,
    users: List[User],
    base_stations: List[BaseStation],
    leos: List[LEOConstellation],
    region: Region,
):
    print("\nStarting RF-Accurate Hybrid Mobility Simulation (Strict 3GPP Admission Control)...")

    duration_s = cfg.simulation.get("duration_s", 86400)
    time_step_s = cfg.simulation.get("time_step_s", 3600)
    time_steps_s = list(range(20 * 3600, duration_s + time_step_s, time_step_s))
    allow_spillover = cfg.simulation.get("allow_spillover", True)

    worker_count = int(cfg.simulation.get("num_workers", _detect_cpus() or 1))
    use_parallel = worker_count > 1

    # [SPATIAL] km-plane projection + per-cell coord array, built once and
    # shipped to workers so each builds its own KD-tree.
    if base_stations:
        lat0 = float(sum(bs.lat for bs in base_stations) / len(base_stations))
    else:
        lat0 = 0.0
    cell_xy = np.array(
        [_project_km(bs.lat, bs.lon, lat0) for bs in base_stations],
        dtype=np.float64) if base_stations else np.zeros((0, 2))
    max_cov_r_km = max((bs.coverage_radius_km for bs in base_stations), default=0.0)
    max_intf_cutoff_m = max((bs.interference_cutoff_m for bs in base_stations), default=0.0)

    hex_to_candidate_towers: Dict[str, List[BaseStation]] = {}
    for bs in base_stations:
        for hex_id in bs.covered_h3_ids:
            hex_to_candidate_towers.setdefault(hex_id, []).append(bs)

    bs_by_id: Dict[Any, BaseStation] = {bs.bs_id: bs for bs in base_stations}

    user_data_export = []
    summary_data = []
    beam_animation_data = []
    user_animation_data = []
    detailed_drop_log = []

    print("\U0001f4c1 Initializing chunked CSV log files...")
    pd.DataFrame(columns=["Hour", "Hour_of_Day", "User_ID", "Lat", "Lon", "State"]).to_csv("user_hourly_states.csv", index=False)
    pd.DataFrame(columns=["Time_s", "Hour", "User_ID", "Lat", "Lon", "Demand_Mbps", "TN_Eval_BS", "TN_Eval_MHz", "TN_SINR_dB", "TN_Reason", "NTN_Eval_Beam", "NTN_Eval_MHz", "NTN_SINR_dB", "NTN_Reason", "TN_S_dBm", "TN_I_dBm", "TN_N_dBm", "TN_NumIntf", "TN_IoverN_dB","Final_State"]).to_csv("detailed_drop_log.csv", index=False)

    g_rx_ue_dbi = cfg.terrestrial.get("g_rx_ue_dbi", 0.0)
    sinr_min_tn = cfg.terrestrial.get("sinr_min_db", -3.0)

    executor = None
    try:
        if use_parallel:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_attachment_worker,
                initargs=(base_stations, g_rx_ue_dbi, lat0,
                          cell_xy, max_cov_r_km, max_intf_cutoff_m),
            )
            print(f"\u2699\ufe0f  PHASE 1 parallelism enabled: {worker_count} worker processes "
                  f"(spatial KD-tree prefilter, {len(base_stations):,} cells).")
        else:
            _init_attachment_worker(base_stations, g_rx_ue_dbi, lat0,
                                    cell_xy, max_cov_r_km, max_intf_cutoff_m)
            print("\u2699\ufe0f  PHASE 1 running serially (num_workers <= 1).")

        for t_s in time_steps_s:
            hour_of_day = (t_s / 3600.0) % 24.0
            absolute_hour = t_s / 3600.0
            total_demand = 0.0
            total_served_tn = 0.0
            unmet_demand_ledger: Dict[str, List[Dict[str, Any]]] = {}

            for bs in base_stations:
                bs.remaining_bandwidth_hz = bs.total_bandwidth_hz
                bs.active_users = 0
                bs.attached_users.clear()

            for u in users:
                u.current_demand = u.get_demand_at_time(hour_of_day)
                total_demand += u.current_demand
                u.served_mbps = 0.0
                u.locked_to_tn = False
                u.coverage_type = "IDLE" if u.current_demand < 0.1 else "DROPPED"

                u.tn_eval_bs = "None"
                u.tn_reason = "N/A"
                u.tn_eval_hz = 0.0
                u.ntn_eval_beam = "None"
                u.ntn_reason = "N/A"
                u.ntn_eval_hz = 0.0

                u.move(hour_of_day, region.h3_resolution)

            # ==================================================
            # PHASE 1: CELL ATTACHMENT  [PARALLEL + SPATIAL]
            # ==================================================
            active_users = [u for u in users if u.current_demand >= 0.1]
            payload = [(u.current_lat, u.current_lon) for u in active_users]

            if active_users:
                if executor is not None:
                    chunk = max(1, len(active_users) // (worker_count * 4))
                    results = executor.map(_evaluate_attachment, payload, chunksize=chunk)
                else:
                    results = map(_evaluate_attachment, payload)

                for u, (best_bs_id, best_sinr, best_spec_eff, best_diag) in zip(active_users, results):
                    u.tn_S_dbm = best_diag["S_dBm"]
                    u.tn_I_dbm = best_diag["I_dBm"]
                    u.tn_N_dbm = best_diag["N_dBm"]
                    u.tn_num_interferers = best_diag["num_interferers"]
                    u.tn_IoverN_db = best_diag["IoverN_dB"]
                    if best_bs_id is not None and best_sinr >= sinr_min_tn:
                        u.tn_sinr_db = best_sinr
                        u.spectral_efficiency = best_spec_eff
                        u.tn_eval_bs = f"BS_{best_bs_id}"
                        bs_by_id[best_bs_id].attached_users.append(u)
                    elif best_bs_id is not None:
                        u.tn_sinr_db = best_sinr
                        u.tn_reason = f"5G SINR too low ({best_sinr:.1f} dB)"
                        u.tn_eval_bs = f"BS_{best_bs_id}"
                    else:
                        u.tn_reason = "No 5G Tower in Geographic Range"

            # ==================================================
            # PHASE 2: MAC SCHEDULING
            # ==================================================
            for bs in base_stations:
                if not bs.attached_users:
                    continue

                for u in bs.attached_users:
                    u.achievable_rate_mbps = (bs.remaining_bandwidth_hz * u.spectral_efficiency) / 1e6
                    u.pf_score = u.achievable_rate_mbps / max(0.1, getattr(u, 'historical_avg_mbps', 0.1))

                bs.attached_users.sort(key=lambda x: x.pf_score, reverse=True)

                for u in bs.attached_users:
                    u.tn_eval_hz = bs.remaining_bandwidth_hz

                    if bs.remaining_bandwidth_hz <= 0:
                        u.tn_reason = "5G Congestion (Tower Empty)"
                        break

                    required_hz = (u.current_demand * 1e6) / u.spectral_efficiency
                    min_qos_hz = (getattr(u, 'qos_min_mbps', 0.1) * 1e6) / u.spectral_efficiency

                    if required_hz <= bs.remaining_bandwidth_hz:
                        bs.remaining_bandwidth_hz -= required_hz
                        u.served_mbps = u.current_demand
                        u.coverage_type = "TN"
                        u.locked_to_tn = True
                        u.tn_reason = "Fully Served"
                    elif bs.remaining_bandwidth_hz >= min_qos_hz:
                        allocated_hz = bs.remaining_bandwidth_hz
                        bs.remaining_bandwidth_hz = 0.0
                        u.served_mbps = (allocated_hz * u.spectral_efficiency) / 1e6
                        u.coverage_type = "TN"
                        u.locked_to_tn = not allow_spillover
                        u.tn_reason = "Partially Served (Congested)"
                    else:
                        u.locked_to_tn = False
                        u.tn_reason = f"5G Bandwidth too low for QoS (Req: {min_qos_hz/1e6:.1f} MHz)"

                    bs.active_users += 1
                    total_served_tn += u.served_mbps
                    u.historical_avg_mbps = (0.8 * getattr(u, 'historical_avg_mbps', 0.1)) + (0.2 * u.served_mbps)

            # ==================================================
            # PHASE 3: SPILLOVER LEDGER BINDING
            # ==================================================
            for u in users:
                unmet = u.current_demand - u.served_mbps
                if unmet > 0.1 and not getattr(u, 'locked_to_tn', False):
                    if u.current_h3_id not in unmet_demand_ledger:
                        unmet_demand_ledger[u.current_h3_id] = []
                    unmet_demand_ledger[u.current_h3_id].append(
                        {"user": u, "unmet_mbps": unmet, "initial_unmet": unmet}
                    )

                if t_s == 0:
                    user_data_export.append({"User_ID": u.user_id, "Demand_Mbps": round(u.current_demand, 2), "H3_Cell": u.current_h3_id})

            leo_total_load = sum(sum(e["unmet_mbps"] for e in u_list) for u_list in unmet_demand_ledger.values())

            # ==================================================
            # PHASE 4: NTN FALLBACK EXECUTION
            # ==================================================
            active_beams = allocate_ntn_beams(cfg, leos, unmet_demand_ledger, t_s)

            for beam in active_beams:
                beam_animation_data.append({
                    "time_s": t_s,
                    "h3_id": beam.target_cell_id,
                    "satellite": beam.satellite_id,
                    "elevation": round(beam.elevation_deg, 1)
                })

            for u_list in unmet_demand_ledger.values():
                for entry in u_list:
                    if entry["unmet_mbps"] < entry["initial_unmet"]:
                        entry["user"].coverage_type = "LEO"
                    elif entry["unmet_mbps"] > 0.1 and entry["user"].coverage_type != "TN":
                        entry["user"].coverage_type = "DROPPED"

            for u in users:
                VISUALISTION_SAMPLING = cfg.simulation.get("visualization_sampling", False)
                VISUALIZATION_SAMPLE_RATE = cfg.simulation.get("visualization_sample_rate", 1)
                if VISUALISTION_SAMPLING:
                    if u.user_id % VISUALIZATION_SAMPLE_RATE == 0:
                        user_animation_data.append({
                            "Hour": f"Hour {absolute_hour:.1f}",
                            "Hour_of_Day": round(hour_of_day, 2),
                            "User_ID": u.user_id,
                            "Lat": u.current_lat,
                            "Lon": u.current_lon,
                            "State": u.coverage_type
                        })
                else:
                    user_animation_data.append({
                        "Hour": f"Hour {absolute_hour:.1f}",
                        "Hour_of_Day": round(hour_of_day, 2),
                        "User_ID": u.user_id,
                        "Lat": u.current_lat,
                        "Lon": u.current_lon,
                        "State": u.coverage_type
                    })

                if u.current_demand > 0.1:
                    detailed_drop_log.append({
                        "Time_s": t_s,
                        "Hour": round(absolute_hour, 2),
                        "Hour_of_Day": round(hour_of_day, 2),
                        "User_ID": u.user_id,
                        "Lat": round(u.current_lat, 4),
                        "Lon": round(u.current_lon, 4),
                        "Demand_Mbps": round(u.current_demand, 2),
                        "TN_Eval_BS": u.tn_eval_bs,
                        "TN_Eval_MHz": round(u.tn_eval_hz / 1e6, 2),
                        "TN_SINR_dB":  round(u.tn_sinr_db, 2),
                        "TN_Reason": u.tn_reason,
                        "NTN_Eval_Beam": u.ntn_eval_beam,
                        "NTN_Eval_MHz": round(u.ntn_eval_hz / 1e6, 2),
                        "NTN_SINR_dB": round(u.ntn_sinr_db, 2),
                        "NTN_Reason": u.ntn_reason,
                        "TN_S_dBm": round(getattr(u, 'tn_S_dbm', float('nan')), 2),
                        "TN_I_dBm": round(getattr(u, 'tn_I_dbm', float('nan')), 2),
                        "TN_N_dBm": round(getattr(u, 'tn_N_dbm', float('nan')), 2),
                        "TN_NumIntf": getattr(u, 'tn_num_interferers', 0),
                        "TN_IoverN_dB": round(getattr(u, 'tn_IoverN_db', float('nan')), 2),
                        "Final_State": u.coverage_type
                    })

            dropped_traffic = sum(sum(e["unmet_mbps"] for e in u_list) for u_list in unmet_demand_ledger.values())
            total_served_ntn = leo_total_load - dropped_traffic

            if user_animation_data:
                pd.DataFrame(user_animation_data).to_csv("user_hourly_states.csv", mode='a', header=False, index=False)
                user_animation_data.clear()

            if detailed_drop_log:
                pd.DataFrame(detailed_drop_log).to_csv("detailed_drop_log.csv", mode='a', header=False, index=False)
                detailed_drop_log.clear()

            summary_data.append({
                "Time_s": t_s, "Hour": round(hour_of_day, 2),
                "Total_Demand_Mbps": round(total_demand, 2),
                "Served_TN_Mbps": round(total_served_tn, 2),
                "Served_NTN_Mbps": round(total_served_ntn, 2),
                "Active_NTN_Beams": len(active_beams),
                "Dropped_Traffic_Mbps": round(dropped_traffic, 2)
            })

            print(f"  [t={t_s:05d}s | {hour_of_day:04.1f}h] Demand: {total_demand:7.1f} | TN Served: {total_served_tn:7.1f} | NTN Served: {total_served_ntn:7.1f} | Dropped: {dropped_traffic:7.1f} Mbps")

    finally:
        if executor is not None:
            executor.shutdown()

    pd.DataFrame(user_data_export).to_csv("users_initial_state.csv", index=False)
    pd.DataFrame(summary_data).to_csv("system_summary_table.csv", index=False)

    print("\n\u2705 Simulation Complete. Generated all export files.")
    return beam_animation_data, user_animation_data