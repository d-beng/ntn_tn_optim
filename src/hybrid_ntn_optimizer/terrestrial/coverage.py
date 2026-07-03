import os
import math
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from omegaconf import DictConfig, OmegaConf
import h3
from threadpoolctl import threadpool_limits

from scipy.spatial import ConvexHull, cKDTree

from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation, DeploymentScenario


# ======================================================================
# SCENARIO CLASSIFICATION — REAL WORLDPOP/H3 DENSITY (people per km^2)
# ----------------------------------------------------------------------
# K-means is kept for tower PLACEMENT (it puts towers where people are),
# but the UMi/UMa/RMa label is no longer derived from cluster geometry
# (which produced a meaningless ~20,000 users/km^2 artifact, because it
# divided real headcounts by tiny sub-km cluster radii).
#
# Instead, each tower inherits the density of the H3 cell it sits in:
#       density = (users whose home is in that H3 cell) / (true H3 cell area)
# H3 cell area is a fixed real geographic area, so this is genuine
# people/km^2 -- directly comparable to real-world thresholds.
#
# Classification (cfg.terrestrial.density_umi / density_uma, defaults shown):
#   density >= 1000 people/km^2  -> UMI  (dense urban core)
#   density >=  400 people/km^2  -> UMA  (urban / suburban)
#   else                         -> RMA  (rural)
#
# Density resolution: cfg.terrestrial.density_h3_resolution (default 7).
#   res 7 hexagon ~ 5.16 km^2  (neighborhood scale -> good for UMi/UMa/RMa)
#   res 6 hexagon ~ 36.1 km^2  (town scale)
#   res 8 hexagon ~ 0.737 km^2 (block scale)
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


def _cluster_radius_km_vec(center: np.ndarray, pts: np.ndarray) -> float:
    """Vectorized max haversine distance (km) from center to assigned points."""
    if pts is None or len(pts) == 0:
        return 0.0
    lat0 = math.radians(float(center[0]))
    lon0 = math.radians(float(center[1]))
    lat = np.radians(pts[:, 0].astype(np.float64))
    lon = np.radians(pts[:, 1].astype(np.float64))
    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    d_km = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(d_km.max())


def _make_radius_boundary(center, radius_km: float, n_points: int = 72) -> List[List[float]]:
    lat = float(center[0]); lon = float(center[1])
    radius_km = max(float(radius_km), 0.05)
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    boundary: List[List[float]] = []
    for i in range(n_points + 1):
        angle = 2.0 * math.pi * i / n_points
        d_lat = (radius_km / 111.0) * math.sin(angle)
        d_lon = (radius_km / (111.0 * cos_lat)) * math.cos(angle)
        boundary.append([lat + d_lat, lon + d_lon])
    return boundary


def _make_membership_boundary(pts: np.ndarray, center, fallback_radius_km: float,
                              hull_sample_cap: int = 4000) -> List[List[float]]:
    if pts is None or len(pts) == 0:
        return _make_radius_boundary(center, max(0.2, min(1.0, fallback_radius_km)), n_points=24)
    sample = pts
    if len(pts) > hull_sample_cap:
        idx = np.random.choice(len(pts), size=hull_sample_cap, replace=False)
        sample = pts[idx]
    if len(sample) >= 3 and ConvexHull is not None:
        try:
            hull = ConvexHull(sample)
            coords = sample[hull.vertices].tolist()
            coords.append(coords[0])
            return [[float(lat), float(lon)] for lat, lon in coords]
        except Exception:
            pass
    return _make_radius_boundary(center, max(0.2, min(1.0, float(fallback_radius_km))), n_points=24)


def _reset_user_runtime_reference(users: List[User]) -> None:
    for u in users:
        u.tn_cell_id = -1
        u.coverage_type = "Unknown"


# ----------------------------------------------------------------------
# REAL DENSITY MAP: people per km^2 per H3 cell, from WorldPop user homes
# ----------------------------------------------------------------------
def _hex_area_km2(res: int) -> float:
    """True average H3 hexagon area in km^2 (version-robust)."""
    if h3 is not None:
        for fn, unit in (("average_hexagon_area", "km^2"), ("hex_area", "km^2")):
            try:
                return float(getattr(h3, fn)(res, unit))
            except Exception:
                continue
    # Fallback constants (km^2) if h3 unavailable: res -> area
    table = {5: 252.903, 6: 36.129, 7: 5.161, 8: 0.737, 9: 0.105}
    return table.get(res, 5.161)


def _build_density_map(all_coords: np.ndarray, dens_res: int) -> Tuple[Dict[str, float], float]:
    """Return {h3_cell -> people/km^2} using real H3 cell area.
    Counts user homes per H3 cell, divides by the true cell area."""
    area_km2 = _hex_area_km2(dens_res)
    counts: Dict[str, int] = {}
    if h3 is None:
        return {}, area_km2
    # latlng_to_cell per point (vectorize via python loop; fast enough at 14.5M
    # only because this runs once; if too slow, sample, but full is fine).
    latlng = h3.latlng_to_cell
    for lat, lon in all_coords:
        c = latlng(float(lat), float(lon), dens_res)
        counts[c] = counts.get(c, 0) + 1
    return {c: n / area_km2 for c, n in counts.items()}, area_km2


def _classify_by_real_density(density: float, cfg) -> str:
    """UMi/UMa/RMa from REAL people/km^2."""
    umi_min = float(_cfg_get(cfg, "terrestrial.density_umi", 1000.0))
    uma_min = float(_cfg_get(cfg, "terrestrial.density_uma", 400.0))
    if density >= umi_min:
        return "UMI"
    elif density >= uma_min:
        return "UMA"
    return "RMA"


# ----------------------------------------------------------------------
# Parallel second-pass worker: one dense discovery zone -> final clusters
# ----------------------------------------------------------------------
def _fit_zone_kmeans(points: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    k = max(1, min(int(k), len(points)))
    if k == 1:
        labels = np.zeros(len(points), dtype=int)
        centers = points.mean(axis=0).reshape(1, 2)
        return labels, centers
    model = KMeans(n_clusters=k, random_state=seed, n_init=3)
    labels = model.fit_predict(points)
    return labels, model.cluster_centers_


def _process_zone(payload):
    """Worker: cluster one dense zone. Returns candidate-tower tuples.
    Classification is done LATER in the main process via the density map,
    so workers only need geometry + size."""
    (zone_coords, zone_center, zone_size, k_for_zone,
     cluster_user_threshold, seed, hull_cap) = payload

    if threadpool_limits is not None:
        with threadpool_limits(limits=1):
            labels, centers = _fit_zone_kmeans(zone_coords, k_for_zone, seed)
    else:
        labels, centers = _fit_zone_kmeans(zone_coords, k_for_zone, seed)

    zone_radius_km = _cluster_radius_km_vec(np.asarray(zone_center), zone_coords)

    results = []
    for cid in range(len(centers)):
        member = np.where(labels == cid)[0]
        size = len(member)
        if size <= cluster_user_threshold:
            continue
        pts = zone_coords[member]
        center = centers[cid]
        raw_radius_km = _cluster_radius_km_vec(center, pts)
        hull = _make_membership_boundary(pts, center, raw_radius_km, hull_cap)
        results.append((
            float(center[0]), float(center[1]),
            float(raw_radius_km), int(size),
            hull, float(zone_radius_km), int(zone_size),
        ))
    return results


# ----------------------------------------------------------------------
# Overlap suppression (unchanged)
# ----------------------------------------------------------------------
def _deoverlap(cands: List[dict], overlap_factor: float) -> List[dict]:
    n = len(cands)
    if n == 0 or cKDTree is None or overlap_factor <= 0:
        return cands
    lats = np.array([c["lat"] for c in cands], dtype=np.float64)
    lons = np.array([c["lon"] for c in cands], dtype=np.float64)
    lat0 = math.radians(float(lats.mean()))
    R = 6371.0088
    x = np.radians(lons) * math.cos(lat0) * R
    y = np.radians(lats) * R
    xy = np.column_stack([x, y])
    tree = cKDTree(xy)
    order = np.argsort([-c["assigned_user_count"] for c in cands])
    suppressed = np.zeros(n, dtype=bool)
    kept: List[int] = []
    for i in order:
        if suppressed[i]:
            continue
        kept.append(i)
        r = max(0.1, cands[i]["coverage_radius_km"] * overlap_factor)
        for j in tree.query_ball_point(xy[i], r):
            if j != i:
                suppressed[j] = True
    return [cands[i] for i in kept]



# ----------------------------------------------------------------------
# RMa GAP BACKFILL
# After the primary (density-classified) towers are placed, some user homes
# fall in the interstitial GAPS between the small UMi/UMa coverage circles.
# We find those uncovered users and drop RMa macro cells over them (via the
# same k-means-per-cluster idea) so the big RMa footprint blankets the gaps
# the small cells miss. RMa is placed ONLY where uncovered users cluster
# (>= min_users), so empty land is never filled.
# ----------------------------------------------------------------------
def _find_uncovered_users(all_coords, candidates, cfg):
    """Return indices of user homes NOT within any placed tower's coverage
    radius. Uses a per-tower KDTree query in the local km plane."""
    if cKDTree is None or not candidates:
        return np.arange(len(all_coords))
    lat0 = math.radians(float(all_coords[:, 0].mean()))
    R = 6371.0088
    ux = np.radians(all_coords[:, 1].astype(np.float64)) * math.cos(lat0) * R
    uy = np.radians(all_coords[:, 0].astype(np.float64)) * R
    covered = np.zeros(len(all_coords), dtype=bool)
    # Group towers by radius so we can do one radius-query per distinct radius.
    tx = np.array([math.radians(c["lon"]) * math.cos(lat0) * R for c in candidates])
    ty = np.array([math.radians(c["lat"]) * R for c in candidates])
    radii = np.array([c["coverage_radius_km"] for c in candidates])
    user_tree = cKDTree(np.column_stack([ux, uy]))
    for r in np.unique(radii):
        sel = np.where(radii == r)[0]
        if len(sel) == 0:
            continue
        tree_t = cKDTree(np.column_stack([tx[sel], ty[sel]]))
        # users within r of ANY tower of this radius
        hit = tree_t.query_ball_point(np.column_stack([ux, uy]), r)
        for i, lst in enumerate(hit):
            if lst:
                covered[i] = True
    return np.where(~covered)[0]



def _backfill_rma(all_coords, uncovered_idx, bs_cfg, cfg, random_seed, density_map=None):
    """GEOMETRIC gap-fill: cover the uncovered (gap) users with RMa cells.

    The backfill's job is COVERAGE, not capacity -- fill the geographic gaps
    between the small UMi/UMa circles. So we tile the gap-users with a hex
    lattice at RMa spacing: every gap-user ends up within one RMa radius of a
    node. The NUMBER of RMa cells emerges purely from how spread out the gaps
    are (tight gaps -> few cells; scattered gaps -> more). No magic number.

    A node becomes an RMa cell if it captures at least one gap-user
    (config terrestrial.rma_backfill_min_users overrides; default 1)."""
    if len(uncovered_idx) == 0:
        return []
    rma_r = float(bs_cfg["RMA"]["coverage_radius_km"])
    packing = float(_cfg_get(cfg, "terrestrial.hex_packing", 0.95))
    min_users = int(_cfg_get(cfg, "terrestrial.rma_backfill_min_users", 1))
    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 7))

    pts = all_coords[uncovered_idx].astype(np.float64)   # lat, lon
    lat0 = math.radians(float(pts[:, 0].mean()))
    R = 6371.0088
    x = np.radians(pts[:, 1]) * math.cos(lat0) * R       # km plane
    y = np.radians(pts[:, 0]) * R

    # Hex lattice spacing that guarantees coverage: d = r*sqrt(3)*packing.
    d = rma_r * math.sqrt(3.0) * packing
    dx = d
    dy = d * math.sqrt(3.0) / 2.0
    x0, y0 = x.min(), y.min()
    row = np.round((y - y0) / dy).astype(np.int64)
    col = np.round((x - x0 - (row & 1) * (dx * 0.5)) / dx).astype(np.int64)
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
        nx, ny = float(node_x[g[0]]), float(node_y[g[0]])
        # node km -> lat/lon
        clat = math.degrees(ny / R)
        clon = math.degrees(nx / (R * math.cos(lat0)))
        cell = h3.latlng_to_cell(clat, clon, dens_res)
        real_dens = float(density_map.get(cell, 0.0)) if density_map else 0.0
        out.append({
            "lat": clat, "lon": clon,
            "raw_radius_km": rma_r, "density": real_dens,
            "assigned_user_count": int(len(g)),
            "membership_boundary": _make_radius_boundary((clat, clon), rma_r, 24),
            "zone_radius_km": rma_r, "zone_size": int(len(g)),
            "scenario_key": "RMA",
            "coverage_radius_km": rma_r,
        })
    print(f"   RMa backfill (geometric): {len(uncovered_idx):,} gap-users "
          f"-> {len(out):,} RMa cells at {d:.2f} km spacing.", flush=True)
    return out



# ----------------------------------------------------------------------
# MOVEMENT-AWARE COORDS (multi-snapshot, no double counting)
# move() is STOCHASTIC: each call maybe sends a user to a random attractor.
# One snapshot is a single noisy draw, so we take SEVERAL snapshots across the
# configured hours. To represent where each user actually spends time WITHOUT
# counting anyone more than once, we reduce each user to ONE representative
# position: the mode (most frequent H3 cell) of their snapshot positions.
# This single point per user reflects their dominant location across movement
# (home if they rarely move, an attractor if they usually go there).
# Config:
#   terrestrial.mobility_hours     : hours to sample (e.g. [8,12,18,20,22])
#   terrestrial.snapshots_per_hour : stochastic draws per hour (default 1)
# ----------------------------------------------------------------------
def _representative_positions(users, cfg, region_res):
    """One position per user = mode of their snapshot cells across hours/draws.
    Each user contributes exactly ONE point -> never double-counted."""
    hours = list(_cfg_get(cfg, "terrestrial.mobility_hours", []))
    if not hours:
        return np.asarray([[u.home_lat, u.home_lon] for u in users], dtype=np.float32)
    draws = int(_cfg_get(cfg, "terrestrial.snapshots_per_hour", 1))
    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 7))
    latlng = h3.latlng_to_cell

    n = len(users)
    # For each user, count how often each (lat,lon) cell appears across snapshots,
    # then pick the most frequent -> their representative position.
    from collections import Counter
    rep = np.empty((n, 2), dtype=np.float32)
    # Accumulate per-user cell -> representative latlon (store a sample latlon per cell).
    counters = [Counter() for _ in range(n)]
    cell_latlon = [dict() for _ in range(n)]
    for h in hours:
        for _ in range(max(1, draws)):
            hf = float(h)
            for i, u in enumerate(users):
                try:
                    u.move(hf, region_res)
                    la, lo = u.current_lat, u.current_lon
                except Exception:
                    la, lo = u.home_lat, u.home_lon
                c = latlng(float(la), float(lo), dens_res)
                counters[i][c] += 1
                if c not in cell_latlon[i]:
                    cell_latlon[i][c] = (la, lo)
    # restore users to home
    for u in users:
        u.current_lat = u.home_lat; u.current_lon = u.home_lon
    # also seed home as a candidate so non-movers resolve to home
    for i, u in enumerate(users):
        if len(counters[i]) == 0:
            rep[i] = (u.home_lat, u.home_lon); continue
        best_cell = counters[i].most_common(1)[0][0]
        la, lo = cell_latlon[i][best_cell]
        rep[i] = (la, lo)
    return rep


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def generate_terrestrial_network(cfg: DictConfig, users: List[User], h3_resolution: int) -> List[BaseStation]:
    """
    Two-pass parallel TN placement, with scenario classification from
    REAL WorldPop/H3 population density (people per km^2).

    Placement (k-means) is unchanged. The only conceptual change: each tower's
    UMi/UMa/RMa label comes from the real population density of the H3 cell it
    sits in, not from cluster geometry. This gives physically meaningful
    densities directly comparable to the 1000 / 400 people/km^2 thresholds.
    """
    print(" [PLACEMENT] Two-pass TN placement (real WorldPop/H3 density classification)...", flush=True)
    if not users:
        print(" No users provided; no terrestrial network generated.", flush=True)
        return []

    random_seed = int(_cfg_get(cfg, "random_seed", 42))
    density_threshold = max(1, int(_cfg_get(cfg, "terrestrial.density_threshold", 5000)))
    cluster_user_threshold = max(1, int(_cfg_get(
        cfg, "terrestrial.min_users_per_tn_cluster",
        _cfg_get(cfg, "terrestrial.users_per_cluster_ratio", 1300))))
    overlap_factor = float(_cfg_get(cfg, "terrestrial.overlap_factor", 1.0))
    hull_cap = int(_cfg_get(cfg, "terrestrial.hull_sample_cap", 4000))
    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 7))

    bs_cfg = _cfg_get(cfg, "terrestrial.scenarios", None)
    if bs_cfg is None:
        raise ValueError("cfg.terrestrial.scenarios is required (UMI/UMA/RMA parameters).")

    n_cores = _detect_cpus()
    _reset_user_runtime_reference(users)
    # Movement-aware placement: use positions at the simulated hour (single
    # snapshot, each user once) if configured; else home positions.
    mob_hours = _cfg_get(cfg, "terrestrial.mobility_hours", [])
    if mob_hours:
        print(f"   Movement-aware placement: representative position per user "
              f"(mode over hours {list(mob_hours)}, "
              f"{_cfg_get(cfg,'terrestrial.snapshots_per_hour',1)} draws/hour; "
              f"each user counted once)...", flush=True)
        all_coords = _representative_positions(users, cfg, h3_resolution)
    else:
        all_coords = np.asarray([[u.home_lat, u.home_lon] for u in users], dtype=np.float32)

    # ------------------------------------------------------------------
    # REAL DENSITY MAP (people/km^2 per H3 cell at dens_res) — computed once
    # ------------------------------------------------------------------
    if h3 is None:
        raise RuntimeError("h3 is required for real-density classification. `pip install h3`.")
    print(f"   Building real density map at H3 res {dens_res} "
          f"(cell area {_hex_area_km2(dens_res):.3f} km^2)...", flush=True)
    density_map, _area = _build_density_map(all_coords, dens_res)
    print(f"   Density map: {len(density_map):,} populated cells.", flush=True)

    # ------------------------------------------------------------------
    # PASS 1 — discovery
    # ------------------------------------------------------------------
    k_discovery = max(1, int(math.ceil(len(users) / density_threshold)))
    k_discovery = min(k_discovery, len(users))
    print(f"   Discovery: {k_discovery:,} zones over {len(users):,} users on {n_cores} cores...", flush=True)

    if k_discovery == 1:
        discovery_labels = np.zeros(len(all_coords), dtype=np.int64)
        discovery_centers = all_coords.mean(axis=0).reshape(1, 2)
    else:
        disc = MiniBatchKMeans(
            n_clusters=k_discovery, batch_size=min(100_000, len(all_coords)),
            n_init=3, max_iter=200, random_state=random_seed)
        if threadpool_limits is not None:
            with threadpool_limits(limits=n_cores):
                discovery_labels = disc.fit_predict(all_coords)
        else:
            discovery_labels = disc.fit_predict(all_coords)
        discovery_centers = disc.cluster_centers_

    order = np.argsort(discovery_labels, kind="stable")
    sorted_labels = discovery_labels[order]
    bounds = np.searchsorted(sorted_labels, np.arange(k_discovery), side="left")
    bounds = np.append(bounds, len(sorted_labels))

    payloads = []
    sparse_user_count = 0
    minimum_accepted_size = cluster_user_threshold + 1
    for zid in range(k_discovery):
        idx = order[bounds[zid]:bounds[zid + 1]]
        zone_size = len(idx)
        if zone_size < density_threshold:
            sparse_user_count += zone_size
            continue
        zone_coords = all_coords[idx]
        k_for_zone = max(1, min(zone_size // minimum_accepted_size, zone_size))
        payloads.append((zone_coords, discovery_centers[zid], zone_size, k_for_zone,
                         cluster_user_threshold, random_seed, hull_cap))

    print(f"   Dense zones: {len(payloads):,} ({sparse_user_count:,} users in sparse zones -> NTN). "
          f"Running second pass in parallel...", flush=True)

    # ------------------------------------------------------------------
    # PASS 2 — parallel per-zone k-means
    # ------------------------------------------------------------------
    raw_candidates = []
    if payloads:
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunk = max(1, len(payloads) // (n_cores * 4))
            for zone_results in executor.map(_process_zone, payloads, chunksize=chunk):
                raw_candidates.extend(zone_results)

    # ------------------------------------------------------------------
    # Classify each tower by REAL density of the H3 cell it sits in
    # ------------------------------------------------------------------
    candidates: List[dict] = []
    for (clat, clon, raw_radius_km, size, hull, zone_radius_km, zone_size) in raw_candidates:
        cell = h3.latlng_to_cell(float(clat), float(clon), dens_res)
        real_density = density_map.get(cell, 0.0)          # people/km^2 (real)
        scenario_key = _classify_by_real_density(real_density, cfg)
        candidates.append({
            "lat": clat, "lon": clon,
            "raw_radius_km": raw_radius_km,
            "density": real_density,
            "assigned_user_count": size,
            "membership_boundary": hull,
            "zone_radius_km": zone_radius_km,
            "zone_size": zone_size,
            "scenario_key": scenario_key,
            "coverage_radius_km": float(bs_cfg[scenario_key]["coverage_radius_km"]),
        })

    n_before = len(candidates)
    candidates = _deoverlap(candidates, overlap_factor)
    n_removed = n_before - len(candidates)

    # ------------------------------------------------------------------
    # RMa GAP BACKFILL: cover users left in the interstitial gaps between the
    # small UMi/UMa cells with RMa macro cells (only where uncovered users
    # cluster; empty land stays uncovered -> NTN).
    # ------------------------------------------------------------------
    if bool(_cfg_get(cfg, "terrestrial.rma_backfill", True)):
        # Backfill covers UNCOVERED USER POSITIONS only. NOTE: we deliberately do
        # NOT try to cover every attractor site. Attractors are PER-USER points
        # with Pareto-distributed jump lengths (see traffic/profiles.py), i.e.
        # ~2-3 personal destinations per user => ~30M distinct points whose tail
        # spreads across the whole province (+ gps_wander ~0.005 deg ~ 550 m).
        # Covering that support would blanket the province with RMa. Rare
        # stochastic landings in remote spots are the NTN's job by hybrid design.
        uncovered_idx = _find_uncovered_users(all_coords, candidates, cfg)
        rma_fill = _backfill_rma(all_coords, uncovered_idx, bs_cfg, cfg,
                                 random_seed, density_map=density_map)
        candidates.extend(rma_fill)

    # ------------------------------------------------------------------
    # Build BaseStation objects (unchanged construction)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # SECTORIZATION (3GPP TR 38.901 macro geometry; TR 36.942 4.2.1 pattern)
    # Macro sites (UMa, RMa) are 3-SECTOR: 3 co-located cells at the same
    # lat/lon with boresight azimuths 0/120/240 deg, each carrying the FULL
    # channel bandwidth (frequency reuse-1 across sectors, directional
    # antennas). Small cells (UMi) are single OMNI cells, as is common for
    # street-level small-cell deployments. Each sector is its own BaseStation
    # with its own bandwidth pool and attached_users, so PHASE2 MAC scheduling
    # needs no change; sectors sharing a site are linked by site_id.
    # Config: terrestrial.sectors_per_macro (default 3), terrestrial.umi_sectors
    # (default 1). Set sectors_per_macro=1 to disable and recover omni behaviour.
    # ------------------------------------------------------------------
    sectors_per_macro = int(_cfg_get(cfg, "terrestrial.sectors_per_macro", 3))
    umi_sectors = int(_cfg_get(cfg, "terrestrial.umi_sectors", 1))

    base_stations: List[BaseStation] = []
    bs_id_counter = 0
    for site_id, c in enumerate(candidates):
        scenario_key = c["scenario_key"]
        sc = bs_cfg[scenario_key]
        center = (c["lat"], c["lon"])
        coverage_boundary = _make_radius_boundary(center, sc["coverage_radius_km"])

        # number of sectors for this site
        n_sec = umi_sectors if scenario_key == "UMI" else sectors_per_macro
        n_sec = max(1, n_sec)
        # boresight azimuths evenly spaced; omni site (n_sec==1) -> azimuth None
        if n_sec == 1:
            azimuths = [None]
        else:
            azimuths = [(360.0 / n_sec) * k for k in range(n_sec)]
        # users split across sectors -> per-sector expected load ~ 1/n_sec
        per_sector_users = int(round(c["assigned_user_count"] / n_sec))

        for az in azimuths:
            bs = BaseStation(
                bs_id=bs_id_counter, lat=float(c["lat"]), lon=float(c["lon"]),
                scenario=DeploymentScenario[scenario_key],
                p_tx_dbm=sc["p_tx_dbm"], g_tx_dbi=sc["g_tx_dbi"],
                carrier_freq_hz=sc["carrier_freq_hz"], total_bandwidth_hz=sc["bandwidth_hz"],
                capacity_mbps=sc["bs_capacity_mbps"], bs_height_m=sc["default_h_bs"],
                shadow_sigma_los_db=sc["shadow_sigma_los_db"], shadow_sigma_nlos_db=sc["shadow_sigma_nlos_db"],
                interference_cutoff_m=sc["interference_cutoff_m"], coverage_radius_km=sc["coverage_radius_km"],
                min_user_dist_m=sc["min_user_dist_m"], use_physical_radius=True)

            # --- sector metadata (consumed by full_pipeline attachment + sinr) ---
            bs.site_id = int(site_id)               # co-located sectors share this
            bs.sector_azimuth_deg = az              # boresight; None => omni (UMi)
            bs.num_sectors = int(n_sec)

            bs.voronoi_boundary = c["membership_boundary"]
            bs.coverage_boundary = coverage_boundary
            bs.assigned_user_count = int(per_sector_users)
            bs.raw_cluster_radius_km = float(c["raw_radius_km"])
            bs.cluster_density = float(c["density"])          # REAL people/km^2
            bs.discovery_radius_km = float(c["zone_radius_km"])
            bs.discovery_cluster_size = int(c["zone_size"])
            bs.area_class = "TN-Service-Area"
            bs.set_resolution(h3_resolution)
            base_stations.append(bs)
            bs_id_counter += 1

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
        " TN placement complete: "
        f"{len(base_stations)} base stations (scenario mix: {mix}); "
        f"{len(payloads)} dense zones; {sparse_user_count:,} users in sparse zones (-> NTN); "
        f"{n_removed} overlapping towers suppressed (overlap_factor={overlap_factor}). "
        f"{dens_summary}.", flush=True)
    return base_stations