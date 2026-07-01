import os
import math
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import cKDTree

try:
    import h3
except Exception:  # pragma: no cover
    h3 = None

from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation, DeploymentScenario

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
    return [[lat + (radius_km / 111.0) * math.sin(2*math.pi*i/n_points),
             lon + (radius_km / (111.0*cos_lat)) * math.cos(2*math.pi*i/n_points)]
            for i in range(n_points + 1)]

def _move_chunk_positions(args):
    users_copy, hours, region_res = args
    out = np.empty((len(users_copy) * len(hours), 2), dtype=np.float64)
    k = 0
    for h in hours:
        hf = float(h)
        for u in users_copy:
            try:
                u.move(hf, region_res)
                out[k, 0] = u.current_lat
                out[k, 1] = u.current_lon
            except Exception:
                out[k, 0] = u.home_lat
                out[k, 1] = u.home_lon
            k += 1
    return out

def _collect_positions(cfg, users: List[User], region_res: int) -> np.ndarray:
    hours = list(_cfg_get(cfg, "terrestrial.mobility_hours", [8, 12, 18, 20, 22]))
    home = np.array([[u.home_lat, u.home_lon] for u in users], dtype=np.float64)
    if not hours:
        return home

    n_cores = _detect_cpus()
    n = len(users)

    if n_cores <= 1 or n < 50_000:
        buf = np.empty((n * len(hours), 2), dtype=np.float64)
        k = 0
        for h in hours:
            hf = float(h)
            for u in users:
                try:
                    u.move(hf, region_res); buf[k, 0] = u.current_lat; buf[k, 1] = u.current_lon
                except Exception:
                    buf[k, 0] = u.home_lat; buf[k, 1] = u.home_lon
                k += 1
        for u in users:
            u.current_lat = u.home_lat; u.current_lon = u.home_lon
        return np.vstack([home, buf])

    n_chunks = max(n_cores, 1)
    bounds = np.linspace(0, n, n_chunks + 1, dtype=np.int64)
    payloads = [(users[bounds[i]:bounds[i + 1]], hours, region_res)
                for i in range(n_chunks) if bounds[i + 1] > bounds[i]]

    moved_parts: List[np.ndarray] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=n_cores) as ex:
        for idx, arr in enumerate(ex.map(_move_chunk_positions, payloads)):
            moved_parts[idx] = arr

    moved = np.vstack(moved_parts)
    return np.vstack([home, moved])

def _build_density_map(coords: np.ndarray, dens_res: int) -> Dict[str, float]:
    if h3 is None:
        return {}
    area = _hex_area_km2(dens_res)
    counts: Dict[str, int] = {}
    latlng = h3.latlng_to_cell
    for lat, lon in coords:
        c = latlng(float(lat), float(lon), dens_res)
        counts[c] = counts.get(c, 0) + 1
    return {c: n / area for c, n in counts.items()}

def _classify(density: float, cfg) -> str:
    umi = float(_cfg_get(cfg, "terrestrial.density_umi", 1000.0))
    uma = float(_cfg_get(cfg, "terrestrial.density_uma", 400.0))
    if density >= umi:
        return "UMI"
    if density >= uma:
        return "UMA"
    return "RMA"

def _grid_nodes_for_tier(px, py, radius_km, packing, min_users):
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
    groups = np.split(order, cut)

    out = []
    for g in groups:
        if len(g) < min_users:
            continue
        cx = float(node_x[g[0]]); cy = float(node_y[g[0]])
        out.append((cx, cy, int(len(g))))
    return out

def generate_terrestrial_network(cfg: DictConfig, users: List[User], h3_resolution: int) -> List[BaseStation]:
    print(" [PLACEMENT] Gap-free hex-grid placement (mobility-aware, node-anchored)...", flush=True)
    if not users:
        print(" No users; no TN.", flush=True)
        return []

    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 7))
    # Restored packing to 0.65 to maintain original overlap density
    packing = float(_cfg_get(cfg, "terrestrial.hex_packing", 0.65))
    # Minimum user threshold to prevent building towers for 1 stray car
    min_users = int(_cfg_get(cfg, "terrestrial.min_users_per_tn_cluster", 10))
    bs_cfg = _cfg_get(cfg, "terrestrial.scenarios", None)

    _reset_user_runtime_reference(users)

    print("   Collecting mobility positions (home + sampled hours)...", flush=True)
    coords = _collect_positions(cfg, users, h3_resolution)
    n_users = len(users)
    samples_per_user = len(coords) // n_users

    print(f"   Building real density map at H3 res {dens_res}...", flush=True)
    raw_map = _build_density_map(coords, dens_res)
    density_map = {c: v / max(1, samples_per_user) for c, v in raw_map.items()}

    latlng = h3.latlng_to_cell
    pos_density = np.array([density_map.get(latlng(float(la), float(lo), dens_res), 0.0) for la, lo in coords])
    
    umi_min = float(_cfg_get(cfg, "terrestrial.density_umi", 1000.0))
    uma_min = float(_cfg_get(cfg, "terrestrial.density_uma", 400.0))
    tier_of_pos = np.where(pos_density >= umi_min, "UMI",
                   np.where(pos_density >= uma_min, "UMA", "RMA"))

    lat0 = float(coords[:, 0].mean())
    ux, uy = _to_xy(coords[:, 0], coords[:, 1], lat0)

    claimed = np.zeros(len(coords), dtype=bool)
    all_towers: List[dict] = []
    
    for tier in ["UMI", "UMA", "RMA"]:
        r = float(bs_cfg[tier]["coverage_radius_km"])
        m = (tier_of_pos == tier) & (~claimed)
        if m.sum() == 0:
            continue
            
        nodes = _grid_nodes_for_tier(ux[m], uy[m], r, packing, min_users)
        node_xy = np.array([(nx, ny) for (nx, ny, _) in nodes]) if nodes else np.zeros((0, 2))
        
        for (nx, ny, cnt) in nodes:
            clat, clon = _to_latlon(nx, ny, lat0)
            cell = latlng(clat, clon, dens_res)
            all_towers.append({
                "lat": clat, "lon": clon, "scenario_key": tier,
                "assigned_user_count": cnt,
                "density": density_map.get(cell, 0.0),
                "coverage_radius_km": r,
            })
            
        if len(node_xy):
            mi = np.where(~claimed)[0]
            if len(mi):
                tree = cKDTree(node_xy)
                dists, _ = tree.query(np.column_stack([ux[mi], uy[mi]]), k=1)
                claimed[mi[dists <= r]] = True

    uncovered = int((~claimed).sum())

    print(f"   Candidate towers generated: {len(all_towers):,}. Running spatial suppression...", flush=True)
    
    # Sort towers by density and user count (keep the best towers first)
    all_towers.sort(key=lambda t: (t["density"], t["assigned_user_count"]), reverse=True)
    
    kept_towers = []
    if all_towers:
        pts = np.array([[t["lat"], t["lon"]] for t in all_towers])
        tree = cKDTree(pts)
        suppressed = np.zeros(len(pts), dtype=bool)
        
        # Physical Site Deduplication: 
        # Only suppress towers that are physically co-located (e.g., within 50 meters).
        # This allows Small Cells (UMI) to exist inside Macro Cells (UMA/RMA) naturally,
        # but prevents building two towers on the exact same physical spot.
        co_location_radius_km = 0.05  # 50 meters
        
        for i in range(len(pts)):
            if suppressed[i]:
                continue
            kept_towers.append(all_towers[i])
            # Find neighbors sharing the same physical site
            neighbors = tree.query_ball_point(pts[i], co_location_radius_km)
            for n in neighbors:
                if n != i:
                    suppressed[n] = True

    print(f"   Suppressed {len(all_towers) - len(kept_towers):,} physically duplicated towers.", flush=True)

    base_stations: List[BaseStation] = []
    for bid, c in enumerate(kept_towers):
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
        bs.area_class = "HexGrid-TN"
        bs.set_resolution(h3_resolution)
        base_stations.append(bs)

    mix = {}
    for bs in base_stations:
        mix[bs.scenario.name] = mix.get(bs.scenario.name, 0) + 1

    print(
        f" TN placement complete: {len(base_stations)} base stations (mix: {mix}); "
        f"Uncovered (sent to NTN): {uncovered:,} samples. "
        f"packing={packing}, min_users={min_users}.",
        flush=True)
        
    return base_stations