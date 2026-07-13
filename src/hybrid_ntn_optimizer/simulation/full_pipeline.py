import os
# [MEM] Pin BLAS/OpenMP to 1 thread per process BEFORE numpy loads. Each of
# the 250 forked PHASE-1 workers otherwise spawns its own ~256-thread OpenBLAS
# pool (stacks + work buffers), which alone can OOM the node. The sbatch
# should ALSO export these (authoritative if numpy was already imported by an
# earlier module):
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import gc
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

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


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
_G_RX_UE_DBI: float = 0.0
_KD = None
_LAT0 = 0.0
_MAX_COV_R_KM = 0.0
_MAX_INTF_CUTOFF_M = 0.0
_TOPK = 6          # candidate cells returned per user (capacity-aware attach)
_R_EARTH_KM = 6371.0088

# [SHARED NETWORK SNAPSHOT — fork-inherited, copy-on-write-safe]
# Workers never receive BaseStation OBJECTS (576k objects x 250 workers OOMed
# the node: CPython refcounts defeat copy-on-write, and initargs pickles a
# full private copy per worker). Instead the main process packs the 13 numeric
# fields PHASE 1 actually reads into ONE float64 array + three int arrays,
# sets them as module globals BEFORE the pool is created, and the KD-tree is
# built ONCE in the main process. Linux fork then shares these read-only
# buffers across all workers at ~60 MB total, regardless of worker count.
# Column layout of _CELLS (float64, shape (N, 13)):
#   0 lat | 1 lon | 2 coverage_radius_km | 3 min_user_dist_m
#   4 interference_cutoff_m | 5 p_tx_dbm | 6 g_tx_dbi | 7 carrier_freq_hz
#   8 bandwidth_hz | 9 bs_height_m | 10 shadow_sigma_los_db
#   11 shadow_sigma_nlos_db | 12 sector_azimuth_deg (NaN = omni)
_CELLS = None          # (N, 13) float64
_SITE_IDS = None       # (N,) int64
_SCEN_CODES = None     # (N,) int16  -> index into _SCEN_ENUMS
_BS_IDS = None         # (N,) int64  -> BaseStation.bs_id (explicit, not row idx)
_SCEN_ENUMS = None     # tuple of DeploymentScenario members
_P = None              # dict of EXPLICIT link-budget parameters (no defaults)


def _project_km(lat, lon, lat0):
    x = math.radians(lon) * math.cos(math.radians(lat0)) * _R_EARTH_KM
    y = math.radians(lat) * _R_EARTH_KM
    return x, y


def _set_worker_network(cells, site_ids, scen_codes, bs_ids, scen_enums,
                        params, g_rx_ue_dbi, lat0, kd_tree,
                        max_cov_r_km, max_intf_cutoff_m):
    """Called in the MAIN process BEFORE the pool is created. Workers inherit
    these module globals via fork (arrays are CoW-shared; the cKDTree is a C
    object whose pages are read-only in workers)."""
    global _CELLS, _SITE_IDS, _SCEN_CODES, _BS_IDS, _SCEN_ENUMS, _P
    global _G_RX_UE_DBI, _LAT0, _KD, _MAX_COV_R_KM, _MAX_INTF_CUTOFF_M
    _CELLS = cells
    _SITE_IDS = site_ids
    _SCEN_CODES = scen_codes
    _BS_IDS = bs_ids
    _SCEN_ENUMS = scen_enums
    _P = params
    _G_RX_UE_DBI = g_rx_ue_dbi
    _LAT0 = lat0
    _KD = kd_tree
    _MAX_COV_R_KM = max_cov_r_km
    _MAX_INTF_CUTOFF_M = max_intf_cutoff_m


def _init_attachment_worker():
    """Pool initializer: verify the fork-inherited globals actually arrived.
    (On a non-fork start method they would be None -> fail loudly, never
    silently compute with missing state.) Also disables the cyclic GC in the
    worker: the attachment loop creates no reference cycles (tuples/lists/
    dicts of floats), refcounting frees everything, and a GC pass in a forked
    child would dirty inherited copy-on-write pages."""
    gc.disable()
    if _CELLS is None or _KD is None and cKDTree is not None:
        raise RuntimeError(
            "PHASE 1 worker started without the shared network snapshot. "
            "The pool must be created AFTER _set_worker_network() in a "
            "fork-based start method (Linux default).")


def _evaluate_attachment(user_pos):
    """Side-effect-free PHASE 1 inner loop over the shared array snapshot.

    Input : (user_lat, user_lon)
    Output: list of up to _TOPK candidates (sinr_db, spec_eff, bs_id, diag),
            sorted by SINR desc; [] if no cell is in geographic range.
    """
    u_lat, u_lon = user_pos
    C = _CELLS
    if C is None:
        raise RuntimeError("worker network snapshot missing")

    # --- SPATIAL PREFILTER: candidate serving cells + candidate interferers ---
    if _KD is not None:
        ux, uy = _project_km(u_lat, u_lon, _LAT0)
        serving_idx = _KD.query_ball_point((ux, uy), _MAX_COV_R_KM)
        if not serving_idx:
            return []
        intf_idx = _KD.query_ball_point((ux, uy), _MAX_INTF_CUTOFF_M / 1000.0)
    else:
        serving_idx = range(len(C))
        intf_idx = list(range(len(C)))

    candidates_out = []
    for si in serving_idx:
        row = C[si]
        d_m = haversine_distance(u_lat, u_lon, row[0], row[1])
        if (d_m / 1000.0) > row[2]:
            continue

        # SECTOR ADMISSION: user must lie inside this sector's wedge (omni passes)
        az = row[12]
        az_v = None if math.isnan(az) else float(az)
        if not in_sector(row[0], row[1], az_v, u_lat, u_lon):
            continue

        d_m = max(d_m, row[3])

        # serving sector antenna gain offset toward the user (<=0 dB; 0 for omni)
        serv_sector_gain = sector_gain_db(row[0], row[1], az_v, u_lat, u_lon)

        # interferers: nearby cells within their own cutoff, excluding the
        # serving cell AND all co-located sectors of the serving site.
        # Each interferer is a tuple of RAW NUMBERS (no objects):
        # (dist_m, scenario, fc_hz, h_bs_m, sigma_los, sigma_nlos,
        #  p_tx_dbm, g_tx_dbi, lat, lon, sector_azimuth_or_None)
        serving_site = _SITE_IDS[si]
        interferers = []
        for oi in intf_idx:
            if oi == si:
                continue
            if _SITE_IDS[oi] == serving_site:
                continue
            orow = C[oi]
            dist = haversine_distance(u_lat, u_lon, orow[0], orow[1])
            if dist <= orow[4]:
                dist = max(dist, orow[3])
                oaz = orow[12]
                interferers.append((
                    dist,
                    _SCEN_ENUMS[_SCEN_CODES[oi]],
                    orow[7], orow[9],
                    orow[10], orow[11],
                    orow[5], orow[6],
                    orow[0], orow[1],
                    None if math.isnan(oaz) else float(oaz),
                ))

        # EVERY link-budget parameter passed EXPLICITLY from _P (built from
        # config in the main process) — nothing left to function defaults.
        sinr_db, capacity_mbps, spec_eff, diag = calculate_tn_sinr_capacity(
            bs_height_m=row[9],
            dist_to_serving_m=d_m,
            interferers=interferers,
            shadow_sigma_los_db=row[10],
            shadow_sigma_nlos_db=row[11],
            scenario=_SCEN_ENUMS[_SCEN_CODES[si]],
            p_tx_dbm=row[5],
            g_tx_dbi=row[6] + serv_sector_gain,
            g_rx_ue_dbi=_G_RX_UE_DBI,
            serving_beamforming_gain_db=_P["serving_beamforming_gain_db"],
            interferer_beamforming_suppression_db=_P["interferer_beamforming_suppression_db"],
            carrier_freq_hz=row[7],
            bandwidth_hz=row[8],
            body_loss_db=_P["body_loss_db"],
            noise_figure_db=(_P["noise_figure_fr2_db"] if row[7] > 24e9
                             else _P["noise_figure_db"]),
            implementation_loss_factor=_P["implementation_loss_factor"],
            ue_height_m=_P["ue_height_m"],
            ue_lat=u_lat,
            ue_lon=u_lon,
        )
        candidates_out.append((sinr_db, spec_eff, int(_BS_IDS[si]), diag))

    if not candidates_out:
        return []
    candidates_out.sort(key=lambda t: t[0], reverse=True)
    return candidates_out[:_TOPK]


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
    time_steps_s = list(range(0 * 3600, duration_s + time_step_s, time_step_s))
    allow_spillover = cfg.simulation.get("allow_spillover", True)

    worker_count = int(cfg.simulation.get("num_workers", _detect_cpus() or 1))
    use_parallel = worker_count > 1

    # [SHARED NETWORK SNAPSHOT] Pack the numeric fields PHASE 1 reads into
    # arrays ONCE; workers fork-inherit them copy-on-write (~60 MB total,
    # independent of worker count). BaseStation OBJECTS never leave the main
    # process (phases 2-4 keep using them here).
    if base_stations:
        lat0 = float(sum(bs.lat for bs in base_stations) / len(base_stations))
    else:
        lat0 = 0.0
    N = len(base_stations)
    cells = np.empty((N, 13), dtype=np.float64)
    site_ids = np.empty(N, dtype=np.int64)
    scen_codes = np.empty(N, dtype=np.int16)
    bs_ids = np.empty(N, dtype=np.int64)
    scen_enums = tuple(DeploymentScenario)
    scen_index = {sc: k for k, sc in enumerate(scen_enums)}
    for i, bs in enumerate(base_stations):
        az = getattr(bs, "sector_azimuth_deg", None)
        cells[i, 0] = bs.lat
        cells[i, 1] = bs.lon
        cells[i, 2] = bs.coverage_radius_km
        cells[i, 3] = bs.min_user_dist_m
        cells[i, 4] = bs.interference_cutoff_m
        cells[i, 5] = bs.p_tx_dbm
        cells[i, 6] = bs.g_tx_dbi
        cells[i, 7] = bs.carrier_freq_hz
        cells[i, 8] = bs.total_bandwidth_hz
        cells[i, 9] = bs.bs_height_m
        cells[i, 10] = bs.shadow_sigma_los_db
        cells[i, 11] = bs.shadow_sigma_nlos_db
        cells[i, 12] = float("nan") if az is None else float(az)
        site_ids[i] = int(getattr(bs, "site_id", -1))
        scen_codes[i] = scen_index[bs.scenario]
        bs_ids[i] = int(bs.bs_id)
    cell_xy = np.column_stack([
        np.radians(cells[:, 1]) * math.cos(math.radians(lat0)) * _R_EARTH_KM,
        np.radians(cells[:, 0]) * _R_EARTH_KM,
    ]) if N else np.zeros((0, 2))
    max_cov_r_km = float(cells[:, 2].max()) if N else 0.0
    max_intf_cutoff_m = float(cells[:, 4].max()) if N else 0.0
    kd_tree = cKDTree(cell_xy) if (cKDTree is not None and N) else None

    # EXPLICIT link-budget parameters — read from config with the SAME values
    # the physics used until now; passed explicitly to every SINR call so no
    # function default is silently relied upon. Override any of them under
    # terrestrial.* in the YAML.
    link_params = {
        "serving_beamforming_gain_db":
            float(cfg.terrestrial.get("serving_beamforming_gain_db", 12.0)),
        "interferer_beamforming_suppression_db":
            float(cfg.terrestrial.get("interferer_beamforming_suppression_db", 12.0)),
        "body_loss_db": float(cfg.terrestrial.get("body_loss_db", 3.0)),
        "noise_figure_db": float(cfg.terrestrial.get("noise_figure_db", 7.0)),
        "noise_figure_fr2_db": float(cfg.terrestrial.get("noise_figure_fr2_db", 10.0)),
        "implementation_loss_factor":
            float(cfg.terrestrial.get("implementation_loss_factor", 0.65)),
        "ue_height_m": float(cfg.terrestrial.get("ue_height_m", 1.5)),
    }
    print(f"   [link] explicit params: {link_params}", flush=True)

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

    # Save base station inventory once so visualisation scripts can read it
    # without re-running placement (lightweight, written once per job).
    if base_stations:
        import pandas as _pd
        _pd.DataFrame([{
            "bs_id": bs.bs_id,
            "site_id": getattr(bs, "site_id", bs.bs_id),
            "lat": bs.lat,
            "lon": bs.lon,
            "scenario": bs.scenario.name,
            "coverage_radius_km": bs.coverage_radius_km,
            "sector_azimuth_deg": getattr(bs, "sector_azimuth_deg", None),
            "num_sectors": getattr(bs, "num_sectors", 1),
            # full RF set -> the network can be RELOADED exactly from this CSV
            # (terrestrial.load_bs_csv), incl. mmWave cells whose enum is UMI
            # but whose RF differs (400 MHz / 28 GHz / 30 dBm).
            "p_tx_dbm": bs.p_tx_dbm,
            "g_tx_dbi": bs.g_tx_dbi,
            "carrier_freq_hz": bs.carrier_freq_hz,
            "bandwidth_hz": bs.total_bandwidth_hz,
            "bs_height_m": bs.bs_height_m,
            "shadow_sigma_los_db": bs.shadow_sigma_los_db,
            "shadow_sigma_nlos_db": bs.shadow_sigma_nlos_db,
            "interference_cutoff_m": bs.interference_cutoff_m,
            "min_user_dist_m": bs.min_user_dist_m,
        } for bs in base_stations]).to_csv("base_stations.csv", index=False)
        print(f"\U0001f4be Saved {len(base_stations):,} base stations to base_stations.csv")

    print("\U0001f4c1 Initializing chunked CSV log files...")
    pd.DataFrame(columns=["Hour", "Hour_of_Day", "User_ID", "Lat", "Lon", "State"]).to_csv("user_hourly_states.csv", index=False)
    pd.DataFrame(columns=["Time_s", "Hour", "User_ID", "Lat", "Lon", "Demand_Mbps", "TN_Eval_BS", "TN_Eval_MHz", "TN_SINR_dB", "TN_Reason", "NTN_Eval_Beam", "NTN_Eval_MHz", "NTN_SINR_dB", "NTN_Reason", "TN_S_dBm", "TN_I_dBm", "TN_N_dBm", "TN_NumIntf", "TN_IoverN_dB","Final_State"]).to_csv("detailed_drop_log.csv", index=False)

    g_rx_ue_dbi = cfg.terrestrial.get("g_rx_ue_dbi", 0.0)
    sinr_min_tn = cfg.terrestrial.get("sinr_min_db", -3.0)

    executor = None
    try:
        # Globals FIRST (main process), THEN the pool: fork-children inherit
        # the arrays/KD-tree copy-on-write. Nothing heavy goes through pickle.
        _set_worker_network(cells, site_ids, scen_codes, bs_ids, scen_enums,
                            link_params, g_rx_ue_dbi, lat0, kd_tree,
                            max_cov_r_km, max_intf_cutoff_m)
        # [MEM] Freeze the parent heap before forking: without this, each
        # worker's cyclic GC walks ALL inherited objects (users, BS objects)
        # and WRITES their GC headers, dirtying every copy-on-write page ->
        # 250 silent copies of a multi-GB heap. freeze() moves existing
        # objects to the permanent generation, which child GC never touches.
        gc.collect()
        gc.freeze()
        if use_parallel:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_attachment_worker,
            )
            print(f"\u2699\ufe0f  PHASE 1 parallelism enabled: {worker_count} worker processes "
                  f"(fork-shared array snapshot, {len(base_stations):,} cells, "
                  f"{cells.nbytes/1e6:.0f} MB shared).")
        else:
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
            active_users = [u for u in users if u.current_demand >= 0.0]
            payload = [(u.current_lat, u.current_lon) for u in active_users]

            if active_users:
                if executor is not None:
                    chunk = max(1, len(active_users) // (worker_count * 4))
                    results = executor.map(_evaluate_attachment, payload, chunksize=chunk)
                else:
                    results = map(_evaluate_attachment, payload)

                # running reserved bandwidth per BS for capacity-aware attach
                reserved_hz = {bs.bs_id: 0.0 for bs in base_stations}
                for u, cand_list in zip(active_users, results):
                    if not cand_list:
                        u.tn_reason = "No 5G Tower in Geographic Range"
                        continue
                    # diagnostics from the BEST-SINR candidate (for logging)
                    b_sinr, b_se, b_id, b_diag = cand_list[0]
                    u.tn_S_dbm = b_diag["S_dBm"]; u.tn_I_dbm = b_diag["I_dBm"]
                    u.tn_N_dbm = b_diag["N_dBm"]
                    u.tn_num_interferers = b_diag["num_interferers"]
                    u.tn_IoverN_db = b_diag["IoverN_dB"]

                    # CAPACITY-AWARE ATTACH: among candidates above the SINR
                    # floor, pick the best-SINR cell that still has room for this
                    # user's demand; fall back to best-SINR cell if all are full
                    # (so the drop is correctly attributed to congestion, not
                    # coverage). This spreads load across sectors/towers instead
                    # of piling everyone onto the single peak-SINR cell.
                    chosen = None
                    for (sinr_db, spec_eff, bs_id, diag) in cand_list:
                        if sinr_db < sinr_min_tn:
                            continue
                        bs = bs_by_id[bs_id]
                        need_hz = (u.current_demand * 1e6) / max(spec_eff, 1e-6)
                        if reserved_hz[bs_id] + need_hz <= bs.total_bandwidth_hz:
                            chosen = (sinr_db, spec_eff, bs_id, need_hz)
                            break
                    if chosen is None:
                        # all reachable cells above floor are full -> attach to
                        # best-SINR one anyway; PHASE 2 will mark it congested.
                        above = [c for c in cand_list if c[0] >= sinr_min_tn]
                        if above:
                            sinr_db, spec_eff, bs_id, _ = above[0]
                            chosen = (sinr_db, spec_eff, bs_id,
                                      (u.current_demand * 1e6) / max(spec_eff, 1e-6))
                    if chosen is None:
                        # nothing above SINR floor
                        u.tn_sinr_db = b_sinr
                        u.tn_reason = f"5G SINR too low ({b_sinr:.1f} dB)"
                        u.tn_eval_bs = f"BS_{b_id}"
                        continue

                    sinr_db, spec_eff, bs_id, need_hz = chosen
                    u.tn_sinr_db = sinr_db
                    u.spectral_efficiency = spec_eff
                    u.tn_eval_bs = f"BS_{bs_id}"
                    reserved_hz[bs_id] += need_hz
                    bs_by_id[bs_id].attached_users.append(u)

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

                queue_cut = False
                for u in bs.attached_users:
                    u.tn_eval_hz = bs.remaining_bandwidth_hz

                    if bs.remaining_bandwidth_hz <= 0 or queue_cut:
                        # Label EVERY remaining queued user (the old `break`
                        # left them as "N/A" -> they showed up as unexplained
                        # "Other" drops). They still spill to NTN as before.
                        u.tn_reason = "5G Congestion (Tower Empty)"
                        u.locked_to_tn = False
                        queue_cut = True
                        continue

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

            # ---- PER-BASE-STATION / PER-SECTOR UTILISATION LOG ----
            # One row per sector cell + rollup per physical site, at hour 20.
            if abs(hour_of_day - 20.0) < 0.01:
                bs_rows = []
                site_agg = {}
                for bs in base_stations:
                    used_hz  = bs.total_bandwidth_hz - bs.remaining_bandwidth_hz
                    served   = sum(getattr(u, "served_mbps", 0.0) for u in bs.attached_users)
                    demand   = sum(u.current_demand for u in bs.attached_users)
                    util_pct = 100.0 * used_hz / max(bs.total_bandwidth_hz, 1.0)
                    bs_rows.append({
                        "site_id": getattr(bs, "site_id", bs.bs_id),
                        "bs_id": bs.bs_id,
                        "scenario": bs.scenario.name,
                        "sector_az_deg": getattr(bs, "sector_azimuth_deg", None),
                        "lat": round(bs.lat, 5), "lon": round(bs.lon, 5),
                        "total_MHz": round(bs.total_bandwidth_hz / 1e6, 2),
                        "used_MHz": round(used_hz / 1e6, 3),
                        "util_pct": round(util_pct, 1),
                        "attached_users": len(bs.attached_users),
                        "demand_Mbps": round(demand, 2),
                        "served_Mbps": round(served, 2),
                    })
                    sid = getattr(bs, "site_id", bs.bs_id)
                    a = site_agg.setdefault(sid, {"scenario": bs.scenario.name,
                        "lat": bs.lat, "lon": bs.lon, "sectors": 0,
                        "total_MHz": 0.0, "used_MHz": 0.0,
                        "attached_users": 0, "demand_Mbps": 0.0, "served_Mbps": 0.0})
                    a["sectors"] += 1
                    a["total_MHz"] += bs.total_bandwidth_hz / 1e6
                    a["used_MHz"]  += used_hz / 1e6
                    a["attached_users"] += len(bs.attached_users)
                    a["demand_Mbps"] += demand
                    a["served_Mbps"] += served
                pd.DataFrame(bs_rows).to_csv("bs_sector_utilisation.csv", index=False)
                site_rows = []
                for sid, a in site_agg.items():
                    site_rows.append({
                        "site_id": sid, "scenario": a["scenario"], "sectors": a["sectors"],
                        "lat": round(a["lat"], 5), "lon": round(a["lon"], 5),
                        "site_total_MHz": round(a["total_MHz"], 2),
                        "site_used_MHz": round(a["used_MHz"], 3),
                        "site_util_pct": round(100.0 * a["used_MHz"] / max(a["total_MHz"], 1e-6), 1),
                        "attached_users": a["attached_users"],
                        "demand_Mbps": round(a["demand_Mbps"], 2),
                        "served_Mbps": round(a["served_Mbps"], 2),
                    })
                pd.DataFrame(site_rows).to_csv("site_utilisation.csv", index=False)

                # ---- SE / SINR MEDIANS (per tier, and per beam-cell) ----
                # Validates that densification did not wreck SINR: if median SE
                # or SINR collapsed vs baseline, the added capacity was fake.
                served_tn = [u for u in users if getattr(u, "coverage_type", "") == "TN"]
                def _med(vals):
                    a = np.asarray([v for v in vals if v is not None and not (isinstance(v,float) and math.isnan(v))])
                    return float(np.median(a)) if len(a) else float("nan")
                # per-tier via serving BS scenario
                tier_of = {}
                for bs in base_stations:
                    tier_of[bs.bs_id] = bs.scenario.name
                by_tier = {"UMI": {"se": [], "sinr": []},
                           "UMA": {"se": [], "sinr": []},
                           "RMA": {"se": [], "sinr": []}}
                for u in served_tn:
                    bid = None
                    ev = getattr(u, "tn_eval_bs", "")
                    if isinstance(ev, str) and ev.startswith("BS_"):
                        try: bid = int(ev[3:])
                        except: bid = None
                    t = tier_of.get(bid)
                    if t in by_tier:
                        by_tier[t]["se"].append(getattr(u, "spectral_efficiency", float("nan")))
                        by_tier[t]["sinr"].append(getattr(u, "tn_sinr_db", float("nan")))
                print("   [se/sinr] served-TN medians by tier:")
                for t in ["UMI", "UMA", "RMA"]:
                    n = len(by_tier[t]["se"])
                    print(f"        {t}: n={n:>9,}  SE={_med(by_tier[t]['se']):5.2f} bps/Hz  "
                          f"SINR={_med(by_tier[t]['sinr']):6.1f} dB", flush=True)
                glob_se = _med([getattr(u,'spectral_efficiency',float('nan')) for u in served_tn])
                glob_sinr = _med([getattr(u,'tn_sinr_db',float('nan')) for u in served_tn])
                print(f"        ALL: n={len(served_tn):>9,}  SE={glob_se:5.2f} bps/Hz  "
                      f"SINR={glob_sinr:6.1f} dB", flush=True)

                # per-beam-cell medians (dense cells only) -> CSV
                cell_se = {}
                for u in served_tn:
                    h = getattr(u, "current_h3_id", None)
                    if h is None: continue
                    cell_se.setdefault(h, {"se": [], "sinr": [], "n": 0})
                    cell_se[h]["se"].append(getattr(u,"spectral_efficiency",float("nan")))
                    cell_se[h]["sinr"].append(getattr(u,"tn_sinr_db",float("nan")))
                    cell_se[h]["n"] += 1
                cell_rows_se = [{"h3_id": h,
                                 "served_users": d["n"],
                                 "median_SE_bps_hz": round(_med(d["se"]), 3),
                                 "median_SINR_dB": round(_med(d["sinr"]), 2)}
                                for h, d in cell_se.items()]
                pd.DataFrame(cell_rows_se).sort_values("served_users", ascending=False)\
                    .to_csv("cell_se_sinr.csv", index=False)
                print(f"   [se/sinr] wrote cell_se_sinr.csv ({len(cell_rows_se):,} cells).", flush=True)
                _n_full = sum(1 for r in bs_rows if r["util_pct"] >= 99.0)
                print(f"   [util] wrote bs_sector_utilisation.csv ({len(bs_rows):,} cells) "
                      f"and site_utilisation.csv ({len(site_rows):,} sites); "
                      f"{_n_full:,} cells at >=99% utilisation "
                      f"(mean cell util {np.mean([r['util_pct'] for r in bs_rows]):.1f}%).",
                      flush=True)

            # ==================================================
            # PHASE 3: SPILLOVER LEDGER BINDING
            # ==================================================
            for u in users:
                unmet = u.current_demand - u.served_mbps
                if unmet > 0.0 and not getattr(u, 'locked_to_tn', False):
                    if u.current_h3_id not in unmet_demand_ledger:
                        unmet_demand_ledger[u.current_h3_id] = []
                    unmet_demand_ledger[u.current_h3_id].append(
                        {"user": u, "unmet_mbps": unmet, "initial_unmet": unmet}
                    )

                if t_s == 0:
                    user_data_export.append({"User_ID": u.user_id, "Demand_Mbps": round(u.current_demand, 2), "H3_Cell": u.current_h3_id})

            leo_total_load = sum(sum(e["unmet_mbps"] for e in u_list) for u_list in unmet_demand_ledger.values())

            # ---- PER-CELL NTN DEMAND vs ONE-BEAM CAPACITY LOG ----
            # For each H3 cell in the spillover ledger: how much demand is asked
            # of NTN, how many users, and whether it exceeds ONE beam's capacity.
            # This is the file that answers "does per-cell overflow exceed a
            # beam?" — the crux of the TN-vs-NTN / add-more-BS debate.
            if abs(hour_of_day - 20.0) < 0.01:
                _ntn_bw   = float(cfg.constellation.get("bandwidth_hz", 300e6))
                _ntn_se   = float(cfg.constellation.get("spectral_eff", 1.77))
                _beam_cap = _ntn_bw * _ntn_se / 1e6      # Mbps one beam can serve
                cell_rows = []
                for h3id, u_list in unmet_demand_ledger.items():
                    dem = sum(e["unmet_mbps"] for e in u_list)
                    cell_rows.append({
                        "h3_id": h3id,
                        "ntn_users": len(u_list),
                        "ntn_demand_Mbps": round(dem, 3),
                        "one_beam_cap_Mbps": round(_beam_cap, 1),
                        "beams_needed": round(dem / max(_beam_cap, 1e-6), 3),
                        "exceeds_one_beam": bool(dem > _beam_cap),
                    })
                _dfc = pd.DataFrame(cell_rows).sort_values("ntn_demand_Mbps", ascending=False)
                _dfc.to_csv("ntn_cell_demand.csv", index=False)
                _n_over = int(_dfc["exceeds_one_beam"].sum()) if len(_dfc) else 0
                _tot    = len(_dfc)
                _sum_over = float(_dfc.loc[_dfc["exceeds_one_beam"], "ntn_demand_Mbps"].sum()) if _n_over else 0.0
                print(f"   [ntn] wrote ntn_cell_demand.csv: {_tot:,} cells need NTN; "
                      f"one-beam cap = {_beam_cap:.0f} Mbps; "
                      f"{_n_over:,} cells ({100*_n_over/max(_tot,1):.1f}%) exceed one beam "
                      f"(these hold {_sum_over/1e6:.3f} Tbps of the spillover); "
                      f"median beams_needed = {_dfc['beams_needed'].median():.2f}, "
                      f"max = {_dfc['beams_needed'].max():.1f}.", flush=True)

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
                    elif entry["unmet_mbps"] > 0.0 and entry["user"].coverage_type != "TN":
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

                if u.current_demand > 0.0:
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
                if cfg.population.get("use_worldpop", False):
                    user_animation_data.clear()

            if detailed_drop_log:
                pd.DataFrame(detailed_drop_log).to_csv("detailed_drop_log.csv", mode='a', header=False, index=False)
                if cfg.population.get("use_worldpop", False):
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