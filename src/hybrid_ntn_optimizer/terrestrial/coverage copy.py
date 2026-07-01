import os
import math
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from omegaconf import DictConfig, OmegaConf

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

try:
    from scipy.spatial import ConvexHull, cKDTree
except Exception:  # pragma: no cover
    ConvexHull = None
    cKDTree = None

from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation, DeploymentScenario


# ======================================================================
# RECOMMENDED PARAMETERS (Ontario, ~14.5M full-population agents)
# ----------------------------------------------------------------------
#   terrestrial.density_threshold:        5000   # coarse discovery bucket size
#   terrestrial.min_users_per_tn_cluster: 1300   # target tower load (~N/1300 towers)
#   terrestrial.overlap_factor:           1.0    # suppress towers within 1x coverage radius
#
# Scenario classification is now DENSITY-BASED (users/km^2), not radius-based,
# because local user density is the physical quantity that distinguishes
# UMi (dense urban) / UMa (urban-suburban) / RMa (rural). This makes UMa the
# large medium-density middle tier instead of a thin radius sliver.
#
# Density thresholds (override per-scenario in cfg.terrestrial.scenarios.*.density_min,
# or globally via cfg.terrestrial.density_umi / density_uma):
#   UMi : density >= ~1000 users/km^2   (dense urban cores)
#   UMa : density >= ~100  users/km^2   (urban / suburban)
#   RMa : density <  ~100  users/km^2   (rural)
# ======================================================================


def _cfg_get(cfg: DictConfig, path: str, default):
    """Safe nested config reader for OmegaConf/DictConfig."""
    value = OmegaConf.select(cfg, path, default=default)
    return default if value is None else value


def _detect_cpus() -> int:
    """CPUs actually allocated to this process (SLURM-aware)."""
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n:
        return int(n)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _cluster_radius_km_vec(center: np.ndarray, pts: np.ndarray) -> float:
    """Vectorized max haversine distance (km) from center to assigned points
    (farthest assigned user — same definition as the original _cluster_radius_km)."""
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


def _effective_radius_km_vec(center: np.ndarray, pts: np.ndarray, pct: float = 90.0) -> float:
    """Percentile radius (km) — robust 'size' of the cluster, used for density.
    The max radius is sensitive to a single far outlier; the 90th percentile gives
    a stable footprint estimate for the density calculation."""
    if pts is None or len(pts) == 0:
        return 0.0
    lat0 = math.radians(float(center[0]))
    lat = np.radians(pts[:, 0].astype(np.float64))
    lon = np.radians(pts[:, 1].astype(np.float64))
    dlat = lat - lat0
    dlon = np.radians(pts[:, 1].astype(np.float64)) - math.radians(float(center[1]))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    d_km = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(np.percentile(d_km, pct))


def _make_radius_boundary(center, radius_km: float, n_points: int = 72) -> List[List[float]]:
    """Circular [lat, lon] polygon for the physical BS coverage radius. (Unchanged.)"""
    lat = float(center[0])
    lon = float(center[1])
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
    """K-Means membership / Voronoi-style display polygon (sampled hull at scale)."""
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
# Scenario classification — NOW DENSITY-BASED (the key enhancement)
# ----------------------------------------------------------------------
def _classify_scenario(assigned_users: int, eff_radius_km: float, bs_cfg, cfg) -> str:
    """Classify UMi / UMa / RMa by LOCAL USER DENSITY (users per km^2).

    density = assigned_users / (pi * eff_radius_km^2)

    Thresholds (resolved with sensible defaults):
      density >= density_umi  -> UMI  (dense urban core)
      density >= density_uma  -> UMA  (urban / suburban)
      else                    -> RMA  (rural)

    Per-scenario overrides: cfg.terrestrial.scenarios.UMI.density_min etc.
    Global overrides:       cfg.terrestrial.density_umi / density_uma.
    """
    area_km2 = max(math.pi * (eff_radius_km ** 2), 1e-3)
    density = assigned_users / area_km2   # users / km^2

    # Resolve thresholds: per-scenario density_min wins, else global, else default.
    umi_min = _cfg_get(cfg, "terrestrial.scenarios.UMI.density_min",
                       _cfg_get(cfg, "terrestrial.density_umi", 1000.0))
    uma_min = _cfg_get(cfg, "terrestrial.scenarios.UMA.density_min",
                       _cfg_get(cfg, "terrestrial.density_uma", 100.0))

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
    """Runs in a worker. Returns lightweight candidate-tower tuples for one zone.

    payload = (zone_coords, zone_center, zone_size, k_for_zone,
               cluster_user_threshold, seed, hull_cap)
    Each tuple: (clat, clon, raw_radius_km, eff_radius_km, size, hull,
                 zone_radius_km, zone_size)
    """
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
        if size <= cluster_user_threshold:           # original strict rule
            continue
        pts = zone_coords[member]
        center = centers[cid]
        raw_radius_km = _cluster_radius_km_vec(center, pts)   # full extent (visual/coverage)
        eff_radius_km = _effective_radius_km_vec(center, pts) # robust radius (density)
        hull = _make_membership_boundary(pts, center, raw_radius_km, hull_cap)
        results.append((
            float(center[0]), float(center[1]),
            float(raw_radius_km), float(eff_radius_km), int(size),
            hull, float(zone_radius_km), int(zone_size),
        ))
    return results


# ----------------------------------------------------------------------
# Overlap suppression: drop towers sitting inside a stronger tower's reach
# ----------------------------------------------------------------------
def _deoverlap(cands: List[dict], overlap_factor: float) -> List[dict]:
    """Greedy non-maximum suppression by distance. Keep the busiest tower,
    suppress any other tower whose center lies within overlap_factor * its
    coverage radius. Tower count becomes geometry-driven, not forced."""
    n = len(cands)
    if n == 0 or cKDTree is None or overlap_factor <= 0:
        return cands
    lats = np.array([c["lat"] for c in cands], dtype=np.float64)
    lons = np.array([c["lon"] for c in cands], dtype=np.float64)
    lat0 = math.radians(float(lats.mean()))
    R = 6371.0088
    x = np.radians(lons) * math.cos(lat0) * R   # local equirectangular km
    y = np.radians(lats) * R
    xy = np.column_stack([x, y])
    tree = cKDTree(xy)
    order = np.argsort([-c["assigned_user_count"] for c in cands])  # busiest first
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
# Main entry
# ----------------------------------------------------------------------
def generate_terrestrial_network(cfg: DictConfig, users: List[User], h3_resolution: int) -> List[BaseStation]:
    """
    Two-pass, parallel TN placement with DENSITY-BASED scenario classification.

    - PASS 1 discovery (MiniBatchKMeans) finds dense vs sparse candidate areas.
    - Discovery zones with < density_threshold users get NO TN (left for NTN).
    - PASS 2 K-Means runs ONCE per dense zone, in parallel across cores.
    - A cluster becomes a base station only if it has MORE than
      min_users_per_tn_cluster users.
    - Scenario (UMi/UMa/RMa) is classified by LOCAL USER DENSITY (users/km^2),
      so UMa is the realistic medium-density middle tier.
    - Overlap suppression removes redundant overlapping towers.
    """
    print(" [KMEANS-PLACEMENT] Two-pass TN placement (parallel, density-classified)...", flush=True)
    if not users:
        print(" No users provided; no terrestrial network generated.", flush=True)
        return []

    random_seed = int(_cfg_get(cfg, "random_seed", 42))
    density_threshold = max(1, int(_cfg_get(cfg, "terrestrial.density_threshold", 5000)))
    cluster_user_threshold = max(1, int(_cfg_get(
        cfg, "terrestrial.min_users_per_tn_cluster",
        _cfg_get(cfg, "terrestrial.users_per_cluster_ratio", 1300),
    )))
    overlap_factor = float(_cfg_get(cfg, "terrestrial.overlap_factor", 1.0))
    hull_cap = int(_cfg_get(cfg, "terrestrial.hull_sample_cap", 4000))

    bs_cfg = _cfg_get(cfg, "terrestrial.scenarios", None)
    if bs_cfg is None:
        raise ValueError("cfg.terrestrial.scenarios is required (UMI/UMA/RMA parameters).")

    n_cores = _detect_cpus()

    all_coords = np.asarray([[u.home_lat, u.home_lon] for u in users], dtype=np.float32)
    _reset_user_runtime_reference(users)

    # ------------------------------------------------------------------
    # PASS 1 — discovery (MiniBatchKMeans so k ~ N/density_threshold scales)
    # ------------------------------------------------------------------
    k_discovery = max(1, int(math.ceil(len(users) / density_threshold)))
    k_discovery = min(k_discovery, len(users))
    print(f"   Discovery: {k_discovery:,} zones over {len(users):,} users "
          f"(~{density_threshold} users/zone) on {n_cores} cores...", flush=True)

    if k_discovery == 1:
        discovery_labels = np.zeros(len(all_coords), dtype=np.int64)
        discovery_centers = all_coords.mean(axis=0).reshape(1, 2)
    else:
        disc = MiniBatchKMeans(
            n_clusters=k_discovery,
            batch_size=min(100_000, len(all_coords)),
            n_init=3, max_iter=200, random_state=random_seed,
        )
        if threadpool_limits is not None:
            with threadpool_limits(limits=n_cores):
                discovery_labels = disc.fit_predict(all_coords)
        else:
            discovery_labels = disc.fit_predict(all_coords)
        discovery_centers = disc.cluster_centers_

    # Group point indices by discovery label once (sort + slice).
    order = np.argsort(discovery_labels, kind="stable")
    sorted_labels = discovery_labels[order]
    bounds = np.searchsorted(sorted_labels, np.arange(k_discovery), side="left")
    bounds = np.append(bounds, len(sorted_labels))

    # ------------------------------------------------------------------
    # Build payloads only for DENSE zones (sparse zones skipped -> NTN)
    # ------------------------------------------------------------------
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
        payloads.append((
            zone_coords, discovery_centers[zid], zone_size, k_for_zone,
            cluster_user_threshold, random_seed, hull_cap,
        ))

    print(f"   Dense zones: {len(payloads):,} "
          f"({sparse_user_count:,} users in sparse zones -> NTN). "
          f"Running second pass in parallel...", flush=True)

    # ------------------------------------------------------------------
    # PASS 2 — one K-Means per dense zone, in parallel across cores
    # ------------------------------------------------------------------
    raw_candidates = []
    if payloads:
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunk = max(1, len(payloads) // (n_cores * 4))
            for zone_results in executor.map(_process_zone, payloads, chunksize=chunk):
                raw_candidates.extend(zone_results)

    # ------------------------------------------------------------------
    # Classify scenario (DENSITY-based) and assemble candidates
    # ------------------------------------------------------------------
    candidates: List[dict] = []
    for (clat, clon, raw_radius_km, eff_radius_km, size, hull, zone_radius_km, zone_size) in raw_candidates:
        scenario_key = _classify_scenario(size, eff_radius_km, bs_cfg, cfg)
        area_km2 = max(math.pi * (eff_radius_km ** 2), 1e-3)
        candidates.append({
            "lat": clat, "lon": clon,
            "raw_radius_km": raw_radius_km,
            "eff_radius_km": eff_radius_km,
            "density": size / area_km2,
            "assigned_user_count": size,
            "membership_boundary": hull,
            "zone_radius_km": zone_radius_km,
            "zone_size": zone_size,
            "scenario_key": scenario_key,
            "coverage_radius_km": float(bs_cfg[scenario_key]["coverage_radius_km"]),
        })

    n_before = len(candidates)

    # ------------------------------------------------------------------
    # Overlap suppression -> emergent, non-overlapping tower set
    # ------------------------------------------------------------------
    candidates = _deoverlap(candidates, overlap_factor)
    n_removed = n_before - len(candidates)

    # ------------------------------------------------------------------
    # Build BaseStation objects (sequential bs_id, unchanged construction)
    # ------------------------------------------------------------------
    base_stations: List[BaseStation] = []
    for bs_id_counter, c in enumerate(candidates):
        scenario_key = c["scenario_key"]
        sc = bs_cfg[scenario_key]
        center = (c["lat"], c["lon"])
        coverage_boundary = _make_radius_boundary(center, sc["coverage_radius_km"])

        bs = BaseStation(
            bs_id=bs_id_counter,
            lat=float(c["lat"]),
            lon=float(c["lon"]),
            scenario=DeploymentScenario[scenario_key],
            p_tx_dbm=sc["p_tx_dbm"],
            g_tx_dbi=sc["g_tx_dbi"],
            carrier_freq_hz=sc["carrier_freq_hz"],
            total_bandwidth_hz=sc["bandwidth_hz"],
            capacity_mbps=sc["bs_capacity_mbps"],
            bs_height_m=sc["default_h_bs"],
            shadow_sigma_los_db=sc["shadow_sigma_los_db"],
            shadow_sigma_nlos_db=sc["shadow_sigma_nlos_db"],
            interference_cutoff_m=sc["interference_cutoff_m"],
            coverage_radius_km=sc["coverage_radius_km"],
            min_user_dist_m=sc["min_user_dist_m"],
            use_physical_radius=True,
        )

        bs.voronoi_boundary = c["membership_boundary"]
        bs.coverage_boundary = coverage_boundary
        bs.assigned_user_count = int(c["assigned_user_count"])
        bs.raw_cluster_radius_km = float(c["raw_radius_km"])
        bs.cluster_density = float(c["density"])          # users/km^2 (diagnostic)
        bs.discovery_radius_km = float(c["zone_radius_km"])
        bs.discovery_cluster_size = int(c["zone_size"])
        bs.area_class = "KMeans-TN-Service-Area"
        bs.set_resolution(h3_resolution)

        base_stations.append(bs)

    # Scenario mix + density summary report.
    mix = {}
    for bs in base_stations:
        mix[bs.scenario.name] = mix.get(bs.scenario.name, 0) + 1
    if base_stations:
        dens = np.array([bs.cluster_density for bs in base_stations])
        dens_summary = (f"density users/km^2: p25={np.percentile(dens,25):.0f} "
                        f"median={np.percentile(dens,50):.0f} p75={np.percentile(dens,75):.0f}")
    else:
        dens_summary = "no towers"

    print(
        " K-Means TN placement complete: "
        f"{len(base_stations)} base stations "
        f"(scenario mix: {mix}); "
        f"{len(payloads)} dense zones; "
        f"{sparse_user_count:,} users in sparse zones (-> NTN); "
        f"{n_removed} overlapping towers suppressed (overlap_factor={overlap_factor}). "
        f"{dens_summary}.",
        flush=True,
    )
    return base_stations