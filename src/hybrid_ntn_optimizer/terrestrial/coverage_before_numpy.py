import os
import math
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from omegaconf import DictConfig, OmegaConf

try:
    import h3
except Exception:  # pragma: no cover
    h3 = None

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
    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 9))

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
    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 9))
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

# ----------------------------------------------------------------------
# CAPACITY-CAPPED BACKFILL (couples TN placement to NTN beam capacity)
# For every NTN beam-cell (H3 at the beam-grid resolution), the users left
# UNCOVERED by TN must not exceed what ONE LEO beam can serve:
#     cap_users = beams_per_cell * (ntn_bw_hz * ntn_se / 1e6) / mbps_per_user
# If a cell's uncovered count is above the cap, RMa towers are added over its
# uncovered users (hex nodes, biggest catch first) until the residual fits in
# a single beam. TN therefore stops exactly where NTN capacity begins, and no
# beam-cell is ever handed more spillover than the beam can carry (in
# expectation over the representative positions).
# Config (terrestrial.ntn_cap.*): beams_per_cell(1), bandwidth_hz(300e6),
#   spectral_eff(1.77), mbps_per_user(0.385).
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# URBAN RMa UNDERLAY (low-band coverage blanket beneath the small-cell core)
# Measured finding: in dense (UMI-classified) zones, "No TN Coverage" drops
# sit 85-200 m outside the nearest UMi edge (the inter-UMi slivers) and
# >600 m outside any RMa -- because tier-claiming removed RMa from urban
# cores, nothing blankets the slivers at sim time. Since RMa is 700 MHz and
# UMi is 3.5 GHz (different carriers, no co-channel interference), an RMa
# underlay beneath the UMi carpet is interference-free and mirrors real
# operator practice (low-band coverage layer under mid-band capacity layer).
# This pass lays a plain hex grid of RMa over ALL dense-core positions
# (covered or not), skipping nodes that already have an RMa nearby.
# Config: terrestrial.urban_rma_underlay (default True).
# ----------------------------------------------------------------------
def _urban_rma_underlay(all_coords, pos_density, candidates, bs_cfg, cfg,
                        density_map=None):
    # DEFAULT OFF: the demand-driven densification now fills dense-zone gaps
    # with UMi/UMa (100 MHz) instead of this 20 MHz RMa blanket, which is the
    # right tool where demand is high. Re-enable (urban_rma_underlay: true) only
    # if you want the low-band safety net beneath the small-cell core.
    if not bool(_cfg_get(cfg, "terrestrial.urban_rma_underlay", False)):
        return []
    umi_min = float(_cfg_get(cfg, "terrestrial.density_umi", 1000.0))
    dense_m = pos_density >= umi_min
    if dense_m.sum() == 0:
        return []
    pts = all_coords[dense_m].astype(np.float64)

    rma_r   = float(bs_cfg["RMA"]["coverage_radius_km"])
    packing = float(_cfg_get(cfg, "terrestrial.hex_packing", 0.95))
    d = rma_r * math.sqrt(3.0) * packing
    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 9))
    R = 6371.0088
    lat0 = math.radians(float(pts[:, 0].mean()))
    x = np.radians(pts[:, 1]) * math.cos(lat0) * R
    y = np.radians(pts[:, 0]) * R

    dx = d; dy = d * math.sqrt(3.0) / 2.0
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

    # existing RMa candidates (skip nodes already blanketed)
    ex = [(c["lat"], c["lon"]) for c in candidates if c.get("scenario_key") == "RMA"]
    ex_tree = None
    if ex and cKDTree is not None:
        ex_pts = np.asarray(ex)
        ex_x = np.radians(ex_pts[:, 1]) * math.cos(lat0) * R
        ex_y = np.radians(ex_pts[:, 0]) * R
        ex_tree = cKDTree(np.column_stack([ex_x, ex_y]))

    latlng = h3.latlng_to_cell
    out = []
    for g in groups:
        nx, ny = float(node_x[g[0]]), float(node_y[g[0]])
        if ex_tree is not None:
            dd, _ = ex_tree.query([nx, ny], k=1)
            if dd <= rma_r * 0.8:      # an RMa already blankets this node
                continue
        clat = math.degrees(ny / R)
        clon = math.degrees(nx / (R * math.cos(lat0)))
        dcell = latlng(clat, clon, dens_res)
        out.append({
            "lat": clat, "lon": clon,
            "raw_radius_km": rma_r,
            "density": float(density_map.get(dcell, 0.0)) if density_map else 0.0,
            "assigned_user_count": int(len(g)),
            "membership_boundary": _make_radius_boundary((clat, clon), rma_r, 24),
            "zone_radius_km": rma_r, "zone_size": int(len(g)),
            "scenario_key": "RMA",
            "coverage_radius_km": rma_r,
        })
    print(f"   Urban RMa underlay (700 MHz blanket under 3.5 GHz small cells): "
          f"{len(out):,} RMa added over the dense core "
          f"(closes 85-200 m inter-UMi slivers; no co-channel cost).", flush=True)
    return out


def _capacity_capped_backfill(all_coords, candidates, bs_cfg, cfg,
                              beam_res, mean_demand_mbps, density_map=None):
    # ALL VALUES READ FROM YOUR EXISTING CONFIG / OBJECTS — no parallel knobs.
    #  - beam bandwidth: the SAME key the beam allocator reads
    #    (cfg.constellation.bandwidth_hz), so the cap is always consistent
    #    with what a beam actually has.
    #  - beams per cell: your allocator model is strict one-beam-per-cell.
    #  - spectral efficiency: from the allocator's own link parameters via a
    #    representative served SINR; overridable at cfg.constellation
    #    (spectral_eff) if you log a measured value.
    #  - per-user demand: measured from the ACTUAL users' busy-hour demand
    #    (mean of get_demand_at_time at the simulated peak hour), not a config
    #    constant — passed in by the caller as mean_demand_mbps.
    bw    = float(_cfg_get(cfg, "constellation.bandwidth_hz", 40e6))
    beams = 1.0   # strict one-beam-per-cell (locked model)
    se_cfg = _cfg_get(cfg, "constellation.spectral_eff", None)
    if se_cfg is not None:
        se = float(se_cfg)
    else:
        # representative served NTN SINR: a few dB above the NTN admission
        # floor (cfg.constellation.sinr_min_db); implementation loss 0.65 as
        # in the link-budget module.
        sinr_floor = float(_cfg_get(cfg, "constellation.sinr_min_db", 0.0))
        rep_sinr_db = sinr_floor + 7.5   # matches the measured served median
        se = 0.65 * math.log2(1.0 + 10.0 ** (rep_sinr_db / 10.0))
    mpu = max(float(mean_demand_mbps), 1e-6)
    cap_users = beams * (bw * se / 1e6) / mpu

    uncovered_idx = _find_uncovered_users(all_coords, candidates, cfg)
    if len(uncovered_idx) == 0:
        print("   Capacity cap: nothing uncovered.", flush=True)
        return []
    pts_all = all_coords[uncovered_idx].astype(np.float64)

    # group uncovered users by NTN beam-cell
    latlng = h3.latlng_to_cell
    cell_members = {}
    for i, (la, lo) in enumerate(pts_all):
        c = latlng(float(la), float(lo), beam_res)
        cell_members.setdefault(c, []).append(i)

    rma_r   = float(bs_cfg["RMA"]["coverage_radius_km"])
    packing = float(_cfg_get(cfg, "terrestrial.hex_packing", 0.95))
    d       = rma_r * math.sqrt(3.0) * packing
    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 9))
    R = 6371.0088

    out = []
    n_over_cells = 0
    n_rescued = 0
    for c, idxs in cell_members.items():
        overflow = len(idxs) - cap_users
        if overflow <= 0:
            continue           # one beam can already carry this cell's leftovers
        n_over_cells += 1
        sub = pts_all[np.asarray(idxs)]

        # local km plane for this cell
        lat0 = math.radians(float(sub[:, 0].mean()))
        x = np.radians(sub[:, 1]) * math.cos(lat0) * R
        y = np.radians(sub[:, 0]) * R

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
        groups = sorted(np.split(order, cut), key=len, reverse=True)

        # add towers (biggest catch first) until residual <= cap
        covered_here = 0
        for g in groups:
            if len(idxs) - covered_here <= cap_users:
                break
            nx, ny = float(node_x[g[0]]), float(node_y[g[0]])
            clat = math.degrees(ny / R)
            clon = math.degrees(nx / (R * math.cos(lat0)))
            dcell = latlng(clat, clon, dens_res)
            real_dens = float(density_map.get(dcell, 0.0)) if density_map else 0.0
            out.append({
                "lat": clat, "lon": clon,
                "raw_radius_km": rma_r, "density": real_dens,
                "assigned_user_count": int(len(g)),
                "membership_boundary": _make_radius_boundary((clat, clon), rma_r, 24),
                "zone_radius_km": rma_r, "zone_size": int(len(g)),
                "scenario_key": "RMA",
                "coverage_radius_km": rma_r,
            })
            covered_here += len(g)
        n_rescued += covered_here

    print(f"   Capacity cap [bw={bw/1e6:.0f} MHz (cfg.constellation), "
          f"se={se:.2f} bps/Hz, demand={mpu:.3f} Mbps/user (measured)]: "
          f"one-beam capacity = {cap_users:.0f} users/cell; "
          f"{n_over_cells:,} beam-cells overflowed -> {len(out):,} RMa towers "
          f"added, {n_rescued:,} users pulled onto TN so residual/cell <= cap.",
          flush=True)
    return out



# ----------------------------------------------------------------------
# DEMAND-DRIVEN, PACKING-CAPPED DENSIFICATION (dense-gap UMi/UMa backfill)
# For each NTN beam-cell whose peak demand exceeds what the currently placed
# TN cells there can serve, ADD small cells (UMi in urban, UMa in suburban)
# until EITHER the cell's TN capacity >= its demand OR the hex-packing limit
# is reached (physical density ceiling). Capacity is estimated with the
# DEGRADED spectral efficiency at the tighter spacing, so densification never
# claims capacity the SINR physics would not deliver. Where even max packing
# cannot meet demand, the residual overflows to NTN (documented saturation),
# and RMa remains only for rural gaps + the urban underlay safety net.
#
# SE estimate mode (config terrestrial.densify_real_sinr, default True):
#   True  -> compute SINR at the proposed inter-site distance via the real
#            3GPP link budget (accurate; slower).
#   False -> analytic SE-vs-ISD curve (fast; approximate, documented model).
# ----------------------------------------------------------------------
def _se_from_isd_analytic(isd_km, tier_r_km):
    """Approximate served spectral efficiency [bps/Hz] as small cells pack.
    As inter-site distance shrinks below ~2r, co-channel interference rises and
    SE falls. Anchored: SE~=6.5 at comfortable spacing (isd>=2r), degrading
    toward ~2.5 as isd -> r (heavy overlap). Monotonic, bounded, documented."""
    if tier_r_km <= 0:
        return 6.5
    ratio = isd_km / tier_r_km          # 2.0 = touching-ish, 1.0 = heavy overlap
    se_hi, se_lo = 6.5, 2.5
    x = max(0.0, min(1.0, (ratio - 1.0) / 1.0))   # 1.0 at ratio>=2, 0 at ratio<=1
    return se_lo + (se_hi - se_lo) * x


def _se_from_isd_real(clat, clon, tier_key, isd_km, bs_cfg, cfg):
    """Compute served SE at a representative UE (cell edge, ~0.5*coverage r)
    with 6 interferers on a hex ring at inter-site distance isd_km, using the
    real 3GPP link budget. Returns spectral efficiency [bps/Hz]."""
    try:
        from hybrid_ntn_optimizer.link_budget.sinr import calculate_tn_sinr_capacity
        from hybrid_ntn_optimizer.models.base_station import BaseStation, DeploymentScenario
    except Exception:
        return _se_from_isd_analytic(isd_km, float(bs_cfg[tier_key]["coverage_radius_km"]))
    sc = bs_cfg[tier_key]
    r = float(sc["coverage_radius_km"])
    ue_d_m = max(0.5 * r * 1000.0, float(sc["min_user_dist_m"]))
    # 6 interferers on a ring at isd_km
    class _I:  # lightweight stand-in with the attrs the sinr fn reads
        pass
    interferers = []
    for k in range(6):
        it = _I()
        it.scenario = DeploymentScenario[tier_key]
        it.carrier_freq_hz = sc["carrier_freq_hz"]; it.bs_height_m = sc["default_h_bs"]
        it.p_tx_dbm = sc["p_tx_dbm"]; it.g_tx_dbi = sc["g_tx_dbi"]
        it.shadow_sigma_los_db = sc["shadow_sigma_los_db"]; it.shadow_sigma_nlos_db = sc["shadow_sigma_nlos_db"]
        it.min_user_dist_m = sc["min_user_dist_m"]; it.lat = clat; it.lon = clon
        it.sector_azimuth_deg = None
        interferers.append((it, isd_km * 1000.0))
    try:
        _, _, se, _ = calculate_tn_sinr_capacity(
            dist_to_serving_m=ue_d_m, interferers=interferers,
            scenario=DeploymentScenario[tier_key],
            p_tx_dbm=sc["p_tx_dbm"], g_tx_dbi=sc["g_tx_dbi"],
            carrier_freq_hz=sc["carrier_freq_hz"], bandwidth_hz=sc["bandwidth_hz"],
            bs_height_m=sc["default_h_bs"],
            shadow_sigma_los_db=sc["shadow_sigma_los_db"],
            shadow_sigma_nlos_db=sc["shadow_sigma_nlos_db"])
        return float(se)
    except Exception:
        return _se_from_isd_analytic(isd_km, r)



# ----------------------------------------------------------------------
# SMALL-CELL GAP FILL (simple, transparent — replaces the old densify logic)
# The user's requested behaviour, verbatim: "search a zone where UMi is
# deployed and based on it decide to fill" — i.e. a NORMAL gap fill:
#   1. take positions in DENSE zones (density >= density_uma);
#   2. check coverage against SMALL CELLS ONLY (UMI / UMA / UMI_MMW) —
#      the 20 MHz RMa blanket is IGNORED, because RMa "covering" a dense
#      gap is what was hiding the gaps from the filler;
#   3. tile the uncovered dense users with UMi (urban) / UMa (suburban)
#      on a gap-free hex lattice; optionally UMI_MMW (400 MHz mmWave) in
#      ultra-dense cores if the scenario exists in config.
# No SE-turnover heuristics: spacing is the tier's own lattice spacing
# (d = r*sqrt(3)*packing), i.e. cells are placed at normal grid distance,
# never packed tighter, so SINR stays at deployment-normal levels.
# Config:
#   terrestrial.small_cell_gap_fill (default True)
#   terrestrial.gap_fill_min_users  (default 20)   min gap-users per new cell
#   terrestrial.density_mmw        (default 3000)  people/km^2 for UMI_MMW
#                                   (used only if scenarios.UMI_MMW exists)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# mmWAVE CAPACITY OVERLAY (400 MHz n257 layer over saturated dense cores)
# mmWave's role in real networks: a CAPACITY overlay on top of the sub-6
# coverage layer, never base coverage (150 m cells cannot give contiguous
# coverage). The small-cell GAP fill only helps users OUTSIDE coverage, so
# it can never reach the covered-but-saturated downtown sites. This pass
# lays a UMI_MMW lattice over ALL users in ultra-dense zones
# (density >= terrestrial.density_mmw), covered or not. The capacity-aware
# attachment then automatically shifts overflow onto the 400 MHz layer.
# Config: terrestrial.mmw_overlay (default True, needs scenarios.UMI_MMW),
#         terrestrial.density_mmw (default 7500 people/km^2 — derived from
#         measured saturation: non-sat UMi p90=6.7k, saturated p25=7.5k).
# ----------------------------------------------------------------------
def _mmw_capacity_overlay(all_coords, pos_density, candidates, bs_cfg, cfg,
                          density_map=None):
    if "UMI_MMW" not in bs_cfg:
        return []
    if not bool(_cfg_get(cfg, "terrestrial.mmw_overlay", True)):
        return []
    mmw_min = float(_cfg_get(cfg, "terrestrial.density_mmw", 3000.0))
    min_users = int(_cfg_get(cfg, "terrestrial.gap_fill_min_users", 20))
    packing = float(_cfg_get(cfg, "terrestrial.hex_packing", 0.95))
    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 9))
    R = 6371.0088
    latlng = h3.latlng_to_cell

    m = pos_density >= mmw_min
    if m.sum() == 0:
        print(f"   mmWave overlay: no zones at density >= {mmw_min:.0f}.", flush=True)
        return []
    pts = all_coords[m].astype(np.float64)

    sc = bs_cfg["UMI_MMW"]
    r = float(sc["coverage_radius_km"])
    d = r * math.sqrt(3.0) * packing          # gap-free lattice at mmW scale
    lat0 = math.radians(float(pts[:, 0].mean()))
    x = np.radians(pts[:, 1]) * math.cos(lat0) * R
    y = np.radians(pts[:, 0]) * R
    dx = d; dy = d * math.sqrt(3.0) / 2.0
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

    # skip nodes already blanketed by an existing mmW cell (idempotence)
    ex = [(c["lat"], c["lon"]) for c in candidates
          if c.get("scenario_key") == "UMI_MMW"]
    ex_tree = None
    if ex and cKDTree is not None:
        exa = np.asarray(ex)
        ex_x = np.radians(exa[:, 1]) * math.cos(lat0) * R
        ex_y = np.radians(exa[:, 0]) * R
        ex_tree = cKDTree(np.column_stack([ex_x, ex_y]))

    out = []
    n_users_covered = 0
    for g in groups:
        if len(g) < min_users:
            continue
        nx, ny = float(node_x[g[0]]), float(node_y[g[0]])
        if ex_tree is not None:
            dd, _ = ex_tree.query([nx, ny], k=1)
            if dd <= r * 0.8:
                continue
        clat = math.degrees(ny / R)
        clon = math.degrees(nx / (R * math.cos(lat0)))
        dcell = latlng(clat, clon, dens_res)
        out.append({
            "lat": clat, "lon": clon,
            "raw_radius_km": r,
            "density": float(density_map.get(dcell, 0.0)) if density_map else mmw_min,
            "assigned_user_count": int(len(g)),
            "membership_boundary": _make_radius_boundary((clat, clon), r, 24),
            "zone_radius_km": r, "zone_size": int(len(g)),
            "scenario_key": "UMI_MMW",
            "coverage_radius_km": r,
        })
        n_users_covered += len(g)
    print(f"   mmWave overlay (400 MHz n257, density >= {mmw_min:.0f}): "
          f"{len(out):,} UMI_MMW cells over {n_users_covered:,} ultra-dense-core "
          f"users (capacity layer over existing coverage).", flush=True)
    return out


def _small_cell_gap_fill(all_coords, pos_density, candidates, bs_cfg, cfg,
                         density_map=None):
    if not bool(_cfg_get(cfg, "terrestrial.small_cell_gap_fill", True)):
        return []
    umi_min = float(_cfg_get(cfg, "terrestrial.density_umi", 1000.0))
    uma_min = float(_cfg_get(cfg, "terrestrial.density_uma", 400.0))
    mmw_min = float(_cfg_get(cfg, "terrestrial.density_mmw", 3000.0))
    min_users = int(_cfg_get(cfg, "terrestrial.gap_fill_min_users", 20))
    packing = float(_cfg_get(cfg, "terrestrial.hex_packing", 0.95))
    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 9))
    has_mmw = "UMI_MMW" in bs_cfg
    R = 6371.0088
    latlng = h3.latlng_to_cell

    dense_m = pos_density >= uma_min
    if dense_m.sum() == 0:
        return []
    dense_idx = np.where(dense_m)[0]
    pts = all_coords[dense_idx].astype(np.float64)
    dens = pos_density[dense_idx]

    # --- coverage check vs SMALL CELLS ONLY (RMa deliberately ignored) ---
    small = [c for c in candidates
             if c.get("scenario_key") in ("UMI", "UMA", "UMI_MMW")]
    lat0 = math.radians(float(pts[:, 0].mean()))
    ux = np.radians(pts[:, 1]) * math.cos(lat0) * R
    uy = np.radians(pts[:, 0]) * R
    covered = np.zeros(len(pts), dtype=bool)
    if small and cKDTree is not None:
        sx = np.array([math.radians(c["lon"]) * math.cos(lat0) * R for c in small])
        sy = np.array([math.radians(c["lat"]) * R for c in small])
        radii = np.array([float(c["coverage_radius_km"]) for c in small])
        upts = np.column_stack([ux, uy])
        for r in np.unique(radii):
            sel = np.where(radii == r)[0]
            tree = cKDTree(np.column_stack([sx[sel], sy[sel]]))
            hit = tree.query_ball_point(upts, float(r))
            for i, lst in enumerate(hit):
                if lst:
                    covered[i] = True
    gap = ~covered
    n_gap = int(gap.sum())
    if n_gap == 0:
        print("   Small-cell gap fill: no dense users outside small-cell "
              "coverage.", flush=True)
        return []

    gx, gy = ux[gap], uy[gap]
    gdens = dens[gap]

    # --- choose tier per gap user by local density ---
    if has_mmw:
        tier_of = np.where(gdens >= mmw_min, "UMI_MMW",
                   np.where(gdens >= umi_min, "UMI", "UMA"))
    else:
        tier_of = np.where(gdens >= umi_min, "UMI", "UMA")

    out = []
    added = {}
    for tier in (["UMI_MMW", "UMI", "UMA"] if has_mmw else ["UMI", "UMA"]):
        m = tier_of == tier
        if m.sum() == 0:
            continue
        sc = bs_cfg[tier]
        r = float(sc["coverage_radius_km"])
        d = r * math.sqrt(3.0) * packing          # gap-free lattice spacing
        tx, ty = gx[m], gy[m]
        dx = d
        dy = d * math.sqrt(3.0) / 2.0
        x0, y0 = tx.min(), ty.min()
        row = np.round((ty - y0) / dy).astype(np.int64)
        col = np.round((tx - x0 - (row & 1) * (dx * 0.5)) / dx).astype(np.int64)
        node_x = x0 + col * dx + (row & 1) * (dx * 0.5)
        node_y = y0 + row * dy
        keys = row * 1_000_003 + col
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        cut = np.where(np.diff(ks) != 0)[0] + 1
        groups = np.split(order, cut)
        n_t = 0
        for g in groups:
            if len(g) < min_users:
                continue
            nx, ny = float(node_x[g[0]]), float(node_y[g[0]])
            clat = math.degrees(ny / R)
            clon = math.degrees(nx / (R * math.cos(lat0)))
            dcell = latlng(clat, clon, dens_res)
            out.append({
                "lat": clat, "lon": clon,
                "raw_radius_km": r,
                "density": float(density_map.get(dcell, 0.0)) if density_map else float(gdens[m].mean()),
                "assigned_user_count": int(len(g)),
                "membership_boundary": _make_radius_boundary((clat, clon), r, 24),
                "zone_radius_km": r, "zone_size": int(len(g)),
                "scenario_key": tier,
                "coverage_radius_km": r,
            })
            n_t += 1
        added[tier] = n_t

    msg = ", ".join(f"+{n} {t}" for t, n in added.items())
    print(f"   Small-cell gap fill (RMa blanket IGNORED): {n_gap:,} dense "
          f"users outside small-cell coverage -> {msg} "
          f"(min {min_users} users/cell, normal lattice spacing).", flush=True)
    return out


def _demand_driven_densify(all_coords, pos_density, candidates, bs_cfg, cfg,
                           beam_res, mean_demand_mbps, density_map=None):
    """Add UMi/UMa cells in overflowing dense beam-cells until demand is met or
    the packing limit is hit, using degraded SE so capacity is never imaginary."""
    if not bool(_cfg_get(cfg, "terrestrial.demand_densify", True)):
        return []
    umi_min = float(_cfg_get(cfg, "terrestrial.density_umi", 1000.0))
    uma_min = float(_cfg_get(cfg, "terrestrial.density_uma", 400.0))
    real_sinr = bool(_cfg_get(cfg, "terrestrial.densify_real_sinr", True))
    packing_min = float(_cfg_get(cfg, "terrestrial.densify_packing_min", 1.0))
    # packing_min = min ISD/r allowed (1.0 = cells may reach heavy overlap).

    latlng = h3.latlng_to_cell
    # group ALL positions by beam-cell, with their density (to pick tier)
    cell_pts = {}
    for i, (la, lo) in enumerate(all_coords):
        c = latlng(float(la), float(lo), beam_res)
        cell_pts.setdefault(c, []).append(i)

    # existing TN capacity per beam-cell (sum over placed cells whose center
    # falls in that beam-cell) — approximate: count candidates per cell * their
    # bw * a nominal SE. We compare demand to this and top up.
    R = 6371.0088
    out = []
    n_cells_densified = 0
    n_added = {"UMI": 0, "UMA": 0}
    for c, idxs in cell_pts.items():
        # peak users in this beam-cell (representative positions) & demand
        n_users = len(idxs)
        demand_mbps = n_users * mean_demand_mbps
        # local density -> which small-cell tier to add
        dens = float(np.mean(pos_density[np.asarray(idxs)]))
        if dens >= umi_min:
            tier = "UMI"
        elif dens >= uma_min:
            tier = "UMA"
        else:
            continue   # not a dense cell -> leave to RMa/rural logic

        sc = bs_cfg[tier]
        r = float(sc["coverage_radius_km"]); bw = float(sc["bandwidth_hz"])

        # existing small cells already placed in this beam-cell
        sub = all_coords[np.asarray(idxs)]
        clat = float(sub[:, 0].mean()); clon = float(sub[:, 1].mean())
        existing = 0
        for cand in candidates:
            if cand.get("scenario_key") == tier and                latlng(cand["lat"], cand["lon"], beam_res) == c:
                existing += 1
        existing = max(existing, 1)

        # cell area for packing limit: how many tier cells physically fit
        cell_area = _hex_area_km2(int(beam_res))
        cell_cover = math.pi * r * r
        max_cells = max(1, int(cell_area / (cell_cover * (packing_min ** 2))))

        # iterate: add cells one at a time, recompute ISD -> SE -> capacity
        n_cells = existing
        served_cap = 0.0
        # interference-free spacing: above ~3x coverage radius, an added cell
        # is too far to interfere, so SE stays at its baseline. Below it, SE
        # degrades. This is the REAL spatial scale -- cells spread across a
        # 253 km^2 beam-cell do NOT all interfere; only near-neighbours do.
        isd_free_km = 3.0 * r
        def _cap(nc):
            if nc <= 0: return 0.0, 6.5
            isd = math.sqrt(cell_area / nc)     # mean inter-site distance
            if isd >= isd_free_km:
                # cells far enough apart -> effectively no mutual interference,
                # SE at baseline (each 100 MHz cell delivers full capacity).
                se = _se_from_isd_real(clat, clon, tier, isd_free_km, bs_cfg, cfg) \
                     if real_sinr else _se_from_isd_analytic(isd_free_km, r)
            else:
                se = _se_from_isd_real(clat, clon, tier, isd, bs_cfg, cfg) \
                     if real_sinr else _se_from_isd_analytic(isd, r)
            return nc * bw * se / 1e6, se
        served_cap, se_now = _cap(n_cells)
        prev_cap = served_cap
        added_here = 0
        while served_cap < demand_mbps and n_cells < max_cells:
            n_cells += 1
            new_cap, se_now = _cap(n_cells)
            if new_cap <= prev_cap:      # SE degraded faster than MHz gained
                n_cells -= 1
                break
            prev_cap = new_cap
            served_cap = new_cap
            added_here += 1

        if added_here <= 0:
            continue
        n_cells_densified += 1
        n_added[tier] += added_here

        # place the added cells on a small hex lattice inside the beam-cell
        lat0 = math.radians(clat)
        x = np.radians(sub[:, 1]) * math.cos(lat0) * R
        y = np.radians(sub[:, 0]) * R
        isd = math.sqrt(cell_area / n_cells)
        dx = isd; dy = isd * math.sqrt(3.0) / 2.0
        x0, y0 = x.min(), y.min()
        row = np.round((y - y0) / dy).astype(np.int64)
        col = np.round((x - x0 - (row & 1) * (dx * 0.5)) / dx).astype(np.int64)
        node_x = x0 + col * dx + (row & 1) * (dx * 0.5)
        node_y = y0 + row * dy
        keys = row * 1_000_003 + col
        seen = set()
        placed = 0
        order = np.argsort([len(idxs)])  # dummy
        uniq = {}
        for k, nx, ny in zip(keys, node_x, node_y):
            if k in uniq: continue
            uniq[k] = (nx, ny)
        for k, (nx, ny) in uniq.items():
            if placed >= added_here: break
            clat2 = math.degrees(ny / R)
            clon2 = math.degrees(nx / (R * math.cos(lat0)))
            dcell = latlng(clat2, clon2, int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 9)))
            out.append({
                "lat": clat2, "lon": clon2,
                "raw_radius_km": r,
                "density": float(density_map.get(dcell, 0.0)) if density_map else dens,
                "assigned_user_count": int(n_users / max(n_cells, 1)),
                "membership_boundary": _make_radius_boundary((clat2, clon2), r, 24),
                "zone_radius_km": r, "zone_size": int(n_users),
                "scenario_key": tier,
                "coverage_radius_km": r,
            })
            placed += 1

    print(f"   Demand densify [{'REAL-SINR' if real_sinr else 'analytic'} SE, "
          f"packing_min={packing_min}]: {n_cells_densified:,} dense beam-cells "
          f"topped up -> +{n_added['UMI']:,} UMi, +{n_added['UMA']:,} UMa "
          f"(stopped at demand-met or SE-degradation/packing limit).", flush=True)
    return out


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
    dens_res = int(_cfg_get(cfg, "terrestrial.density_h3_resolution", 9))

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

        # SECOND PASS — capacity coupling: whatever is STILL uncovered must fit
        # within one NTN beam per beam-cell; add RMa where a cell overflows.
        # measure the real per-user busy-hour demand from the users themselves
        _sample = users[:: max(1, len(users) // 100_000)]   # ~100k sample
        _mean_demand = float(np.mean([u.get_demand_at_time(20.0) for u in _sample]))
        cap_fill = _capacity_capped_backfill(all_coords, candidates, bs_cfg,
                                             cfg, h3_resolution, _mean_demand,
                                             density_map=density_map)
        candidates.extend(cap_fill)

        # THIRD PASS — urban RMa underlay: 700 MHz blanket beneath the dense
        # 3.5 GHz small-cell core, closing the 85-200 m inter-UMi slivers that
        # stochastic users land in (measured from the hour-20 drop map).
        _latlng = h3.latlng_to_cell
        pos_density = np.array([density_map.get(_latlng(float(la), float(lo), dens_res), 0.0)
                                for la, lo in all_coords])
        under_fill = _urban_rma_underlay(all_coords, pos_density, candidates,
                                         bs_cfg, cfg, density_map=density_map)
        candidates.extend(under_fill)

        # FOURTH PASS — demand-driven, packing-capped densification: add UMi/UMa
        # in overflowing dense beam-cells until demand is met OR packing/SE limit
        # is hit. Uses degraded SE so added capacity is physically real, not
        # imaginary. RMa is NOT used here (20 MHz too small for dense demand).
        gap_fill = _small_cell_gap_fill(all_coords, pos_density, candidates,
                                        bs_cfg, cfg, density_map=density_map)
        candidates.extend(gap_fill)

        # FIFTH PASS — mmWave capacity overlay: 400 MHz n257 lattice over the
        # ultra-dense cores (covered or not); capacity-aware attachment shifts
        # overflow onto it automatically.
        mmw_fill = _mmw_capacity_overlay(all_coords, pos_density, candidates,
                                         bs_cfg, cfg, density_map=density_map)
        candidates.extend(mmw_fill)

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
    umi_sectors = int(_cfg_get(cfg, "terrestrial.umi_sectors", 3))  # 3GPP UMi
    #   calibration layout is 3 sectors/site; set to 1 in config for omni.

    base_stations: List[BaseStation] = []
    bs_id_counter = 0
    for site_id, c in enumerate(candidates):
        scenario_key = c["scenario_key"]
        sc = bs_cfg[scenario_key]
        center = (c["lat"], c["lon"])
        coverage_boundary = _make_radius_boundary(center, sc["coverage_radius_km"])

        # UMI_MMW (28 GHz n257) uses the UMi propagation scenario -- TR 38.901
        # UMi-Street Canyon is valid 0.5-100 GHz -- with its own RF params
        # (400 MHz, 30 dBm, 23 dBi array) coming from bs_cfg[scenario_key].
        enum_key = "UMI" if scenario_key == "UMI_MMW" else scenario_key

        # number of sectors for this site (mmW treated like UMi small cells)
        n_sec = umi_sectors if scenario_key.startswith("UMI") else sectors_per_macro
        n_sec = max(1, n_sec)
        # boresight azimuths evenly spaced; omni site (n_sec==1) -> azimuth None
        if n_sec == 1:
            azimuths = [None]
        else:
            # 3GPP TR 38.901 calibration azimuths: for a 3-sector site the
            # boresights are 30/150/270 deg (30 deg offset from the hex axes).
            offset = 30.0 if n_sec == 3 else 0.0
            azimuths = [(360.0 / n_sec) * k + offset for k in range(n_sec)]
        # users split across sectors -> per-sector expected load ~ 1/n_sec
        per_sector_users = int(round(c["assigned_user_count"] / n_sec))

        for az in azimuths:
            bs = BaseStation(
                bs_id=bs_id_counter, lat=float(c["lat"]), lon=float(c["lon"]),
                scenario=DeploymentScenario[enum_key],
                p_tx_dbm=sc["p_tx_dbm"], g_tx_dbi=sc["g_tx_dbi"],
                carrier_freq_hz=sc["carrier_freq_hz"], total_bandwidth_hz=sc["bandwidth_hz"],
                capacity_mbps=sc["bs_capacity_mbps"], bs_height_m=sc["default_h_bs"],
                shadow_sigma_los_db=sc["shadow_sigma_los_db"], shadow_sigma_nlos_db=sc["shadow_sigma_nlos_db"],
                interference_cutoff_m=sc["interference_cutoff_m"], coverage_radius_km=sc["coverage_radius_km"],
                min_user_dist_m=sc["min_user_dist_m"], use_physical_radius=True)

            # --- sector metadata (consumed by full_pipeline attachment + sinr) ---
            bs.site_id = int(site_id)               # co-located sectors share this
            bs.sector_azimuth_deg = az              # boresight; None => omni (UMi)

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