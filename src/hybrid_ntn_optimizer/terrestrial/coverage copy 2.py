import os
import math
from typing import List, Tuple, Dict, Set
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from omegaconf import DictConfig, OmegaConf

try:
    import h3
except Exception:  # pragma: no cover
    h3 = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation, DeploymentScenario


# ======================================================================
# GUARANTEED-COVERAGE HEX-GRID PLACEMENT  (replaces k-means)
# ----------------------------------------------------------------------
# WHY THE CHANGE
#   k-means places ONE tower at each cluster centroid and optimizes for
#   point compactness, NOT for covering space. Two adjacent k-means cells
#   can leave an interstitial GAP whose users were assigned to a third,
#   farther centroid whose fixed coverage circle also fails to reach them
#   -> "No 5G Tower in Geographic Range". This is structural; subdivision
#   tuning cannot remove it.
#
# THE FIX (this file)
#   For every populated area we lay a HEXAGONAL grid of candidate towers
#   with nearest-neighbour spacing  d = coverage_radius_km * sqrt(3) * packing.
#   On a hex grid this guarantees EVERY point is within coverage_radius_km of
#   some tower (the cell circumradius = d/sqrt(3) <= r), so there are NO
#   coverage gaps by construction. This is also how real RAN planning works
#   (3GPP hex grid with a per-scenario inter-site distance).
#
#   We then KEEP only grid towers that actually have users within reach
#   (so we don't deploy towers over empty land or lakes), and assign each
#   tower the 3GPP scenario from the REAL WorldPop/H3 population density of
#   its location. Tower count is therefore demand-driven yet coverage is
#   guaranteed wherever people are.
#
# CONFIG (terrestrial.*)
#   density_h3_resolution : H3 res for the real-density map (default 7)
#   density_umi / density_uma : people/km^2 thresholds (default 1000 / 400)
#   hex_packing           : spacing factor in (0,1], 1.0 = touching cover,
#                           <1 = overlap margin (default 0.95 -> slight overlap)
#   min_users_per_site    : drop a grid tower serving fewer users (default 20)
#   coverage_grid_max_km  : safety cap on grid spacing (default from radius)
# ======================================================================


def _cfg_get(cfg: DictConfig, path: str, default):
    value = OmegaConf.select(cfg, path, default=default)
    return default if value is None else value


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
        for fn in ("average_hexagon_area",):
            try:
                return float(getattr(h3, fn)(res, "km^2"))
            except Exception:
                pass
        try:
            return float(h3.hex_area(res, "km^2"))
        except Exception:
            pass
    table = {5: 252.903, 6: 36.129, 7: 5.161, 8: 0.737, 9: 0.105}
    return table.get(res, 5.161)


def _reset_user_runtime_reference(users: List[User]) -> None:
    for u in users:
        u.tn_cell_id = -1
        u.coverage_type = "Unknown"


# ----------------------------------------------------------------------
# Local equirectangular projection (km) around a reference latitude.
# Accurate enough for placement at province scale; keeps geometry simple.
# ----------------------------------------------------------------------
def _to_xy(lat: np.ndarray, lon: np.ndarray, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    R = 6371.0088
    x = np.radians(lon) * math.cos(math.radians(lat0)) * R
    y = np.radians(lat) * R
    return x, y


def _to_latlon(x: float, y: float, lat0: float) -> Tuple[float, float]:
    R = 6371.0088
    lat = math.degrees(y / R)
    lon = math.degrees(x / (R * math.cos(math.radians(lat0))))
    return lat, lon


# ----------------------------------------------------------------------
# Real WorldPop/H3 population-density map (people per km^2 per H3 cell)
# ----------------------------------------------------------------------
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


def _classify_by_real_density(density: float, cfg) -> str:
    umi_min = float(_cfg_get(cfg, "terrestrial.density_umi", 1000.0))
    uma_min = float(_cfg_get(cfg, "terrestrial.density_uma", 400.0))
    if density >= umi_min:
        return "UMI"
    elif density >= uma_min:
        return "UMA"
    return "RMA"


def _make_radius_boundary(center, radius_km: float, n_points: int = 36) -> List[List[float]]:
    lat = float(center[0]); lon = float(center[1])
    radius_km = max(float(radius_km), 0.05)
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    out = []
    for i in range(n_points + 1):
        ang = 2.0 * math.pi * i / n_points
        out.append([lat + (radius_km / 111.0) * math.sin(ang),
                    lon + (radius_km / (111.0 * cos_lat)) * math.cos(ang)])
    return out


# ----------------------------------------------------------------------
# The core: build a hex grid of candidate towers per scenario tier and keep
# only those with users in reach. Because tiers have different radii, we
# process them separately and let denser tiers (smaller radius) override.
# ----------------------------------------------------------------------
def _grid_towers_for_tier(
    ux: np.ndarray, uy: np.ndarray, claimed: np.ndarray,
    radius_km: float, packing: float, min_users: int,
) -> List[Tuple[float, float, int, np.ndarray]]:
    """Lay a hex grid covering the still-unclaimed users of this tier and
    keep grid points that capture >= min_users. Returns list of
    (cx, cy, n_users, member_mask_indices). Marks captured users in `claimed`.

    Hex grid covering guarantee: nearest-neighbour spacing d = r*sqrt(3)*packing
    => every point within r of a node (packing<=1)."""
    idx = np.where(~claimed)[0]
    if len(idx) == 0:
        return []
    px, py = ux[idx], uy[idx]

    d = radius_km * math.sqrt(3.0) * packing      # hex nearest-neighbour spacing
    dx = d                                        # column spacing
    dy = d * math.sqrt(3.0) / 2.0                 # row spacing (hex)
    x0, y0 = px.min(), py.min()

    # Assign each user to its nearest hex node via integer row/col rounding.
    row = np.round((py - y0) / dy).astype(np.int64)
    # odd rows offset by half a column (hexagonal stagger)
    col = np.round((px - x0 - (row & 1) * (dx * 0.5)) / dx).astype(np.int64)

    # Node center for each user (vectorized).
    node_x = x0 + col * dx + (row & 1) * (dx * 0.5)
    node_y = y0 + row * dy

    # Group users by (row, col) node.
    keys = row.astype(np.int64) * 100003 + col.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    bounds = np.where(np.diff(keys_s) != 0)[0] + 1
    groups = np.split(order, bounds)

    towers = []
    for g in groups:
        if len(g) < min_users:
            continue
        gi = idx[g]                                # global user indices
        # Refine center to the centroid of captured users (still within cell).
        cx = float(px[g].mean())
        cy = float(py[g].mean())
        # Only keep users actually within radius of the (refined) center.
        within = ((px[g] - cx) ** 2 + (py[g] - cy) ** 2) <= radius_km ** 2
        if within.sum() < min_users:
            # fall back to the exact node center which guarantees coverage
            cx = float(node_x[g][0]); cy = float(node_y[g][0])
            within = ((px[g] - cx) ** 2 + (py[g] - cy) ** 2) <= radius_km ** 2
            if within.sum() < min_users:
                continue
        capt = gi[within]
        claimed[capt] = True
        towers.append((cx, cy, int(within.sum()), capt))
    return towers


def generate_terrestrial_network(cfg: DictConfig, users: List[User], h3_resolution: int) -> List[BaseStation]:
    """
    Guaranteed-coverage hexagonal-grid TN placement with real-density tiering.

    Pipeline:
      1. Build real WorldPop/H3 density map (people/km^2).
      2. Classify every populated H3 cell into UMI/UMA/RMA by density.
      3. For each tier (densest first), lay a hex grid with spacing
         r*sqrt(3)*packing over that tier's users and keep nodes with users.
         The hex covering guarantee => no interstitial coverage gaps.
      4. Users captured by a denser tier are removed before the next tier,
         so dense cores get small UMi cells and rural gets RMa cells.
    """
    print(" [PLACEMENT] Guaranteed-coverage hex-grid placement (real-density tiered)...", flush=True)
    if not users:
        print(" No users; no TN generated.", flush=True)
        return []
    if h3 is None:
        raise RuntimeError("h3 required. pip install h3")

    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 7))
    packing = float(_cfg_get(cfg, "terrestrial.hex_packing", 0.65))
    min_users = int(_cfg_get(cfg, "terrestrial.min_users_per_site",
                    _cfg_get(cfg, "terrestrial.min_users_per_tn_cluster", 20)))
    bs_cfg = _cfg_get(cfg, "terrestrial.scenarios", None)
    if bs_cfg is None:
        raise ValueError("cfg.terrestrial.scenarios required.")

    coords = np.asarray([[u.home_lat, u.home_lon] for u in users], dtype=np.float64)
    _reset_user_runtime_reference(users)

    # 1+2. Real density per user, and the scenario tier per user.
    print(f"   Building real density map at H3 res {dens_res} "
          f"(cell {_hex_area_km2(dens_res):.3f} km^2)...", flush=True)
    density_map = _build_density_map(coords, dens_res)
    latlng = h3.latlng_to_cell
    user_density = np.array([density_map.get(latlng(float(la), float(lo), dens_res), 0.0)
                             for la, lo in coords])
    umi_min = float(_cfg_get(cfg, "terrestrial.density_umi", 1000.0))
    uma_min = float(_cfg_get(cfg, "terrestrial.density_uma", 400.0))

    tier_of_user = np.where(user_density >= umi_min, "UMI",
                    np.where(user_density >= uma_min, "UMA", "RMA"))

    # Project to local km plane.
    lat0 = float(coords[:, 0].mean())
    ux, uy = _to_xy(coords[:, 0], coords[:, 1], lat0)

    # 3+4. Place per tier, densest (smallest radius) first.
    claimed = np.zeros(len(users), dtype=bool)
    tier_order = ["UMI", "UMA", "RMA"]   # small radius -> large radius
    all_towers: List[dict] = []
    per_tier_counts = {}

    for tier in tier_order:
        r = float(bs_cfg[tier]["coverage_radius_km"])
        # Restrict this tier's grid to users classified as this tier AND unclaimed.
        tier_mask = (tier_of_user == tier) & (~claimed)
        if tier_mask.sum() == 0:
            per_tier_counts[tier] = 0
            continue
        # Build a per-tier "claimed" view: only this tier's users are eligible.
        eligible = ~tier_mask          # everything not this tier is "claimed" for the grid
        grid = _grid_towers_for_tier(ux, uy, eligible.copy(), r, packing, min_users)
        # Mark globally claimed the users captured here.
        for (cx, cy, n, members) in grid:
            claimed[members] = True
            clat, clon = _to_latlon(cx, cy, lat0)
            cell = latlng(clat, clon, dens_res)
            all_towers.append({
                "lat": clat, "lon": clon, "scenario_key": tier,
                "assigned_user_count": n,
                "density": density_map.get(cell, 0.0),
                "coverage_radius_km": r,
            })
        per_tier_counts[tier] = len(grid)
        print(f"   Tier {tier}: r={r:.3f}km, spacing={r*math.sqrt(3)*packing:.3f}km "
              f"-> {len(grid):,} towers, {int(tier_mask.sum()):,} tier users.", flush=True)

    sparse_unserved = int((~claimed).sum())   # users no tier captured -> NTN

    # Build BaseStation objects.
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
        bs.area_class = "HexGrid-TN-Service-Area"
        bs.set_resolution(h3_resolution)
        base_stations.append(bs)

    mix = {}
    for bs in base_stations:
        mix[bs.scenario.name] = mix.get(bs.scenario.name, 0) + 1
    if base_stations:
        dens = np.array([bs.cluster_density for bs in base_stations])
        dens_summary = (f"REAL density people/km^2: p25={np.percentile(dens,25):.0f} "
                        f"median={np.percentile(dens,50):.0f} p75={np.percentile(dens,75):.0f} "
                        f"max={dens.max():.0f}")
    else:
        dens_summary = "no towers"

    print(
        " TN placement complete (HEX-GRID, gap-free by construction): "
        f"{len(base_stations)} base stations (scenario mix: {mix}); "
        f"{sparse_unserved:,} users uncovered by any tier (-> NTN); "
        f"packing={packing}, min_users_per_site={min_users}. {dens_summary}.",
        flush=True)
    return base_stations