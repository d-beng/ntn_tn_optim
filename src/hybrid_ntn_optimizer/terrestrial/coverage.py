import os
import math
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from omegaconf import DictConfig, OmegaConf
import h3

from scipy.spatial import cKDTree
from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation, DeploymentScenario


# ======================================================================
# REALISTIC TWO-LAYER TN PLACEMENT (coverage layer + capacity layer)
# ----------------------------------------------------------------------
# Mirrors how real operators build networks:
#
#   CAPACITY LAYER (UMi/UMa small cells): placed in dense areas for throughput.
#   COVERAGE LAYER (RMa macro cells): backfills populated areas the small cells
#       did not fully cover -- but ONLY where users are actually present. Empty
#       wilderness is never filled (that is NTN's job).
#
# PIPELINE
#   1. Per-hour SNAPSHOTS (each user at ONE position per hour; no double count).
#   2. PEAK-hour density map (busy-hour headcount per H3 cell / area).
#   3. Classify locations by real density: UMi >= density_umi, UMa >= density_uma.
#      (These are the CAPACITY tiers; everything else is candidate RMa coverage.)
#   4. Place the capacity tiers as gap-free hex grids (node-anchored). Mark every
#      user-position they cover.
#   5. Place the RMa COVERAGE layer ONLY over user-positions still uncovered AND
#      dense enough to warrant a tower (>= min_users_per_site users at a node).
#      This backfills populated gaps/fringes, never empty land.
#
# WHY THIS MATCHES THE TARGET MAP
#   TN appears only where people are (snapshots + min_users), concentrated in the
#   dense south/corridors; empty north stays bare; RMa fills the ragged populated
#   fringe the small-cell grid missed, so there are no populated coverage holes.
#
# CONFIG (terrestrial.*)
#   density_h3_resolution : H3 res for density map (default 7 = neighbourhood)
#   density_umi/density_uma : people/km^2 thresholds (default 1000/400)
#   hex_packing           : lattice spacing factor (default 0.65 -> gap-free overlap)
#   min_users_per_site    : min users at a node to place a tower (default 25)
#   mobility_hours        : hours to snapshot (default [8,12,18,20,22]); [] = home
#   rma_backfill          : enable RMa coverage layer (default True)
# ======================================================================


def _cfg_get(cfg, path, default):
    v = OmegaConf.select(cfg, path, default=default)
    return default if v is None else v


def _detect_cpus() -> int:
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n:
        return int(n)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _hex_area_km2(res: int) -> float:
    if h3 is not None:
        try:
            return float(h3.average_hexagon_area(res, "km^2"))
        except Exception:
            try:
                return float(h3.hex_area(res, "km^2"))
            except Exception:
                pass
    return {5: 252.903, 6: 36.129, 7: 5.161, 8: 0.737, 9: 0.105}.get(res, 5.161)


def _reset_user_runtime_reference(users: List[User]) -> None:
    for u in users:
        u.tn_cell_id = -1
        u.coverage_type = "Unknown"


def _to_xy(lat, lon, lat0):
    R = 6371.0088
    return (np.radians(lon) * math.cos(math.radians(lat0)) * R,
            np.radians(lat) * R)


def _to_latlon(x, y, lat0):
    R = 6371.0088
    return (math.degrees(y / R),
            math.degrees(x / (R * math.cos(math.radians(lat0)))))


def _make_radius_boundary(center, radius_km, n_points=36):
    lat = float(center[0]); lon = float(center[1])
    radius_km = max(float(radius_km), 0.05)
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    return [[lat + (radius_km / 111.0) * math.sin(2 * math.pi * i / n_points),
             lon + (radius_km / (111.0 * cos_lat)) * math.cos(2 * math.pi * i / n_points)]
            for i in range(n_points + 1)]


# ----------------------------------------------------------------------
# Per-hour snapshots (parallel; workers mutate only pickled copies, return coords)
# ----------------------------------------------------------------------
def _snapshot_worker(args):
    users_copy, hour, region_res = args
    out = np.empty((len(users_copy), 2), dtype=np.float64)
    hf = float(hour)
    for i, u in enumerate(users_copy):
        try:
            u.move(hf, region_res); out[i, 0] = u.current_lat; out[i, 1] = u.current_lon
        except Exception:
            out[i, 0] = u.home_lat; out[i, 1] = u.home_lon
    return out


def _collect_snapshots(cfg, users, region_res):
    """List of (N,2) arrays, one per sampled hour (+ home). Each user at ONE
    position per snapshot -> never double-counted."""
    hours = list(_cfg_get(cfg, "terrestrial.mobility_hours", [8, 12, 18, 20, 22]))
    home = np.array([[u.home_lat, u.home_lon] for u in users], dtype=np.float64)
    snaps = [home]
    if not hours:
        return snaps

    n_cores = _detect_cpus()
    n = len(users)
    if n_cores <= 1 or n < 50_000:
        for h in hours:
            buf = np.empty((n, 2), dtype=np.float64); hf = float(h)
            for i, u in enumerate(users):
                try:
                    u.move(hf, region_res); buf[i, 0] = u.current_lat; buf[i, 1] = u.current_lon
                except Exception:
                    buf[i, 0] = u.home_lat; buf[i, 1] = u.home_lon
            snaps.append(buf)
        for u in users:
            u.current_lat = u.home_lat; u.current_lon = u.home_lon
        return snaps

    n_chunks = max(n_cores, 1)
    bounds = np.linspace(0, n, n_chunks + 1, dtype=np.int64)
    for h in hours:
        payloads = [(users[bounds[i]:bounds[i + 1]], h, region_res)
                    for i in range(n_chunks) if bounds[i + 1] > bounds[i]]
        parts = [None] * len(payloads)
        with ProcessPoolExecutor(max_workers=n_cores) as ex:
            for idx, arr in enumerate(ex.map(_snapshot_worker, payloads)):
                parts[idx] = arr
        snaps.append(np.vstack(parts))
    return snaps


def _build_peak_density_map(snapshots, dens_res):
    """Per H3 cell: MAX users present in any single snapshot (busy hour) / area."""
    if h3 is None:
        return {}
    area = _hex_area_km2(dens_res)
    latlng = h3.latlng_to_cell
    peak = {}
    for snap in snapshots:
        this_hour = {}
        for lat, lon in snap:
            c = latlng(float(lat), float(lon), dens_res)
            this_hour[c] = this_hour.get(c, 0) + 1
        for c, k in this_hour.items():
            if k > peak.get(c, 0):
                peak[c] = k
    return {c: k / area for c, k in peak.items()}


# ----------------------------------------------------------------------
# Gap-free hex lattice: towers anchored EXACTLY on lattice nodes.
# Returns [(cx, cy, count)] for nodes with >= min_users points.
# ----------------------------------------------------------------------
def _grid_nodes(px, py, radius_km, packing, min_users):
    if len(px) == 0:
        return []
    d = radius_km * math.sqrt(3.0) * packing
    dx = d
    dy = d * math.sqrt(3.0) / 2.0
    x0, y0 = px.min(), py.min()
    row = np.round((py - y0) / dy).astype(np.int64)
    col = np.round((px - x0 - (row & 1) * (dx * 0.5)) / dx).astype(np.int64)
    node_x = x0 + col * dx + (row & 1) * (dx * 0.5)
    node_y = y0 + row * dy
    keys = row * 1_000_003 + col
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    cut = np.where(np.diff(ks) != 0)[0] + 1
    out = []
    for g in np.split(order, cut):
        if len(g) < min_users:
            continue
        out.append((float(node_x[g[0]]), float(node_y[g[0]]), int(len(g))))
    return out


def _mark_covered(claimed, ux, uy, node_xy, radius_km):
    """Mark positions within radius_km of any node in node_xy as claimed."""
    if not len(node_xy) or cKDTree is None:
        return
    idx = np.where(~claimed)[0]
    if not len(idx):
        return
    tree = cKDTree(node_xy)
    dist, _ = tree.query(np.column_stack([ux[idx], uy[idx]]), k=1)
    claimed[idx[dist <= radius_km]] = True


def generate_terrestrial_network(cfg: DictConfig, users: List[User], h3_resolution: int) -> List[BaseStation]:
    print(" [PLACEMENT] Two-layer TN: small-cell capacity + RMa coverage backfill...", flush=True)
    if not users:
        print(" No users; no TN.", flush=True)
        return []
    if h3 is None:
        raise RuntimeError("h3 required.")

    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 7))
    packing = float(_cfg_get(cfg, "terrestrial.hex_packing", 0.65))
    min_users = int(_cfg_get(cfg, "terrestrial.min_users_per_site", 25))
    rma_backfill = bool(_cfg_get(cfg, "terrestrial.rma_backfill", True))
    bs_cfg = _cfg_get(cfg, "terrestrial.scenarios", None)
    if bs_cfg is None:
        raise ValueError("cfg.terrestrial.scenarios required.")

    _reset_user_runtime_reference(users)

    # 1. Snapshots + 2. peak density.
    print("   Collecting per-hour mobility snapshots...", flush=True)
    snapshots = _collect_snapshots(cfg, users, h3_resolution)
    print(f"   {len(snapshots)} snapshots x {len(users):,} users.", flush=True)
    print(f"   Building PEAK-hour density map at H3 res {dens_res} "
          f"(cell {_hex_area_km2(dens_res):.3f} km^2)...", flush=True)
    density_map = _build_peak_density_map(snapshots, dens_res)

    # All snapshot positions used for grid geometry (coverage where users go).
    coords = np.vstack(snapshots)
    lat0 = float(coords[:, 0].mean())
    ux, uy = _to_xy(coords[:, 0], coords[:, 1], lat0)

    latlng = h3.latlng_to_cell
    pos_density = np.array([density_map.get(latlng(float(la), float(lo), dens_res), 0.0)
                            for la, lo in coords])
    umi_min = float(_cfg_get(cfg, "terrestrial.density_umi", 1000.0))
    uma_min = float(_cfg_get(cfg, "terrestrial.density_uma", 400.0))

    claimed = np.zeros(len(coords), dtype=bool)
    all_towers: List[dict] = []

    # 3+4. CAPACITY LAYER: UMi then UMa, gap-free grids over their density class.
    for tier, thresh in (("UMI", umi_min), ("UMA", uma_min)):
        r = float(bs_cfg[tier]["coverage_radius_km"])
        if tier == "UMI":
            m = (pos_density >= umi_min) & (~claimed)
        else:  # UMA: density in [uma_min, umi_min)
            m = (pos_density >= uma_min) & (pos_density < umi_min) & (~claimed)
        if m.sum() == 0:
            print(f"   Tier {tier}: no positions.", flush=True)
            continue
        nodes = _grid_nodes(ux[m], uy[m], r, packing, min_users)
        node_xy = np.array([(nx, ny) for (nx, ny, _) in nodes]) if nodes else np.zeros((0, 2))
        for (nx, ny, cnt) in nodes:
            clat, clon = _to_latlon(nx, ny, lat0)
            all_towers.append({"lat": clat, "lon": clon, "scenario_key": tier,
                               "assigned_user_count": cnt,
                               "density": density_map.get(latlng(clat, clon, dens_res), 0.0),
                               "coverage_radius_km": r})
        _mark_covered(claimed, ux, uy, node_xy, r)
        print(f"   Capacity {tier}: r={r:.3f}km spacing={r*math.sqrt(3)*packing:.3f}km "
              f"-> {len(nodes):,} towers.", flush=True)

    # 5. COVERAGE LAYER: RMa over user-positions STILL uncovered (populated gaps
    #    and fringes) -- but only where >= min_users cluster, so empty land is
    #    never filled. This is the macro coverage blanket for populated areas.
    if rma_backfill:
        r = float(bs_cfg["RMA"]["coverage_radius_km"])
        m = ~claimed  # every still-uncovered user-position (any density)
        if m.sum() > 0:
            nodes = _grid_nodes(ux[m], uy[m], r, packing, min_users)
            node_xy = np.array([(nx, ny) for (nx, ny, _) in nodes]) if nodes else np.zeros((0, 2))
            for (nx, ny, cnt) in nodes:
                clat, clon = _to_latlon(nx, ny, lat0)
                all_towers.append({"lat": clat, "lon": clon, "scenario_key": "RMA",
                                   "assigned_user_count": cnt,
                                   "density": density_map.get(latlng(clat, clon, dens_res), 0.0),
                                   "coverage_radius_km": r})
            _mark_covered(claimed, ux, uy, node_xy, r)
            print(f"   Coverage RMA: r={r:.3f}km spacing={r*math.sqrt(3)*packing:.3f}km "
                  f"-> {len(nodes):,} towers (populated backfill only).", flush=True)

    uncovered = int((~claimed).sum())

    # Build BaseStations.
    base_stations: List[BaseStation] = []
    for bid, c in enumerate(all_towers):
        sc = bs_cfg[c["scenario_key"]]
        bs = BaseStation(
            bs_id=bid, lat=float(c["lat"]), lon=float(c["lon"]),
            scenario=DeploymentScenario[c["scenario_key"]],
            p_tx_dbm=sc["p_tx_dbm"], g_tx_dbi=sc["g_tx_dbi"],
            carrier_freq_hz=sc["carrier_freq_hz"], total_bandwidth_hz=sc["bandwidth_hz"],
            capacity_mbps=sc["bs_capacity_mbps"], bs_height_m=sc["default_h_bs"],
            shadow_sigma_los_db=sc["shadow_sigma_los_db"], shadow_sigma_nlos_db=sc["shadow_sigma_nlos_db"],
            interference_cutoff_m=sc["interference_cutoff_m"], coverage_radius_km=sc["coverage_radius_km"],
            min_user_dist_m=sc["min_user_dist_m"], use_physical_radius=True)
        bs.voronoi_boundary = _make_radius_boundary((c["lat"], c["lon"]), c["coverage_radius_km"], 24)
        bs.coverage_boundary = _make_radius_boundary((c["lat"], c["lon"]), c["coverage_radius_km"], 36)
        bs.assigned_user_count = int(c["assigned_user_count"])
        bs.raw_cluster_radius_km = float(c["coverage_radius_km"])
        bs.cluster_density = float(c["density"])
        bs.area_class = "TwoLayer-TN"
        bs.set_resolution(h3_resolution)
        base_stations.append(bs)

    mix = {}
    for bs in base_stations:
        mix[bs.scenario.name] = mix.get(bs.scenario.name, 0) + 1
    if base_stations:
        dens = np.array([bs.cluster_density for bs in base_stations])
        ds = (f"REAL density people/km^2: p25={np.percentile(dens,25):.0f} "
              f"median={np.percentile(dens,50):.0f} p75={np.percentile(dens,75):.0f} "
              f"max={dens.max():.0f}")
    else:
        ds = "no towers"

    print(
        " TN placement complete (TWO-LAYER: capacity small-cells + RMa coverage backfill): "
        f"{len(base_stations)} base stations (mix: {mix}); "
        f"{uncovered:,} user-positions left to NTN; "
        f"packing={packing}, min_users_per_site={min_users}, res={dens_res}, "
        f"mobility_hours={_cfg_get(cfg,'terrestrial.mobility_hours',[8,12,18,20,22])}. {ds}.",
        flush=True)
    return base_stations