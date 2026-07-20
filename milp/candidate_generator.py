#!/usr/bin/env python3
"""
candidate_generator.py
======================
Day-1 module: everything the per-hex MILP needs, generated automatically
from user positions (no manual site lists).

Produces, for ONE H3 res-5 hex (+ halo):
  candidates : per-tier triangular lattices over POPULATED area only,
               spacing delta_t = R_t*sqrt(3)*rho_cand  (rho_cand ~0.3 gives
               ~<100 m positional resolution for UMi -> information-
               equivalent to the 100 m WorldPop grid)
  demand_pts : users aggregated to H3 res-`agg_res` cells (centroid + Mbps)
  eligibility: N(u) = candidates within R_t and SE >= se_min, pruned to the
               K best (by SE) per demand point
  conflicts  : same-band minimum inter-site-distance pairs
               y_j + y_k <= 1  for ||p_j-p_k|| < d_min = R_t*sqrt(3)*rho_dep
               (rho_dep ~0.95 = physical deployment packing; grounded in the
               measured SINR collapse at ISD < ~1.0 R)

All physics knobs are EXPLICIT parameters -- nothing hidden in defaults.
The SE function is pluggable so the cluster version can call the real
TR 38.901 link budget from sinr.py.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

R_EARTH_KM = 6371.0088

# ---------------------------------------------------------------------------
# Tier definitions (values = your 5g_base.yaml; all overridable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tier:
    name: str
    freq_hz: float
    bw_hz: float
    radius_km: float
    p_tx_dbm: float
    g_tx_dbi: float
    h_bs_m: float
    cost: float                 # deployment cost (relative units)
    density_min: float          # people/km^2 to be a candidate zone

DEFAULT_TIERS = (
    Tier("RMA",     700e6,  20e6, 2.847, 46.0, 15.0, 35.0, cost=2.0, density_min=0.0),
    Tier("UMA",     3.5e9, 100e6, 0.474, 46.0, 17.0, 25.0, cost=2.5, density_min=400.0),
    Tier("UMI",     3.5e9, 100e6, 0.311, 38.0, 10.0, 10.0, cost=1.0, density_min=1000.0),
    Tier("UMI_MMW",  28e9, 400e6, 0.150, 30.0, 23.0, 10.0, cost=1.2, density_min=7500.0),
)

def band_of(tier: Tier) -> str:
    if tier.freq_hz < 1e9:   return "low"
    if tier.freq_hz < 7e9:   return "mid"
    return "mmw"

# ---------------------------------------------------------------------------
# Pluggable spectral-efficiency model (replace with real sinr.py on cluster)
# ---------------------------------------------------------------------------
def se_default(dist_km: float, tier: Tier,
               noise_figure_db: float = 7.0,
               noise_figure_fr2_db: float = 10.0,
               body_loss_db: float = 3.0,
               serving_bf_gain_db: float = 12.0,
               impl_loss: float = 0.65,
               interference_margin_db: float = 6.0,
               se_cap: float = 9.6) -> float:
    """Nominal-interference SE (bps/Hz): log-distance ABG-style pathloss +
    Shannon with implementation loss. Interference frozen as a margin
    (Assumption 1); the simulator recomputes the true value a posteriori."""
    d_m = max(dist_km * 1000.0, 10.0)
    f_ghz = tier.freq_hz / 1e9
    # 3GPP-like urban pathloss slope (UMi/UMa NLOS ~ 3.5-3.9 exponent style)
    pl_db = 32.4 + 21.0 * math.log10(d_m) + 20.0 * math.log10(f_ghz) + 7.8
    nf = noise_figure_fr2_db if tier.freq_hz > 24e9 else noise_figure_db
    noise_dbm = -174.0 + 10.0 * math.log10(tier.bw_hz) + nf
    p_rx = (tier.p_tx_dbm + tier.g_tx_dbi + serving_bf_gain_db
            - body_loss_db - pl_db)
    sinr_db = p_rx - noise_dbm - interference_margin_db
    se = impl_loss * math.log2(1.0 + 10.0 ** (sinr_db / 10.0))
    return float(min(max(se, 0.0), se_cap))

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def project_km(lat, lon, lat0):
    x = np.radians(np.asarray(lon)) * math.cos(math.radians(lat0)) * R_EARTH_KM
    y = np.radians(np.asarray(lat)) * R_EARTH_KM
    return x, y

def unproject(x, y, lat0):
    lat = np.degrees(np.asarray(y) / R_EARTH_KM)
    lon = np.degrees(np.asarray(x) / (R_EARTH_KM * math.cos(math.radians(lat0))))
    return lat, lon

def tri_lattice(x, y, spacing_km):
    """Snap points to a triangular lattice; return unique node coords and the
    per-node captured point count. Nodes exist only where points exist ->
    candidates only over populated area, never on empty land/water."""
    dx = spacing_km
    dy = spacing_km * math.sqrt(3.0) / 2.0
    x0, y0 = x.min(), y.min()
    row = np.round((y - y0) / dy).astype(np.int64)
    col = np.round((x - x0 - (row & 1) * (dx * 0.5)) / dx).astype(np.int64)
    node_x = x0 + col * dx + (row & 1) * (dx * 0.5)
    node_y = y0 + row * dy
    key = row * 1_000_003 + col
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cut = np.where(np.diff(ks) != 0)[0] + 1
    groups = np.split(order, cut)
    nx = np.array([node_x[g[0]] for g in groups])
    ny = np.array([node_y[g[0]] for g in groups])
    cnt = np.array([len(g) for g in groups])
    return nx, ny, cnt

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    # candidates
    cand_xy: np.ndarray        # (J,2) km-plane
    cand_tier: np.ndarray      # (J,) int -> index into tiers
    cand_cost: np.ndarray      # (J,)
    cand_owner_hex: list       # (J,) h3 id owning each candidate
    tiers: tuple
    lat0: float
    # demand
    dem_xy: np.ndarray         # (U,2)
    dem_mbps: np.ndarray       # (U,)
    dem_hex: list              # (U,) h3 res-5 id of each demand point
    # eligibility (sparse): lists per demand point
    elig_j: list               # [ [j,...] per u ]
    elig_se: list              # [ [se,...] per u ]
    elig_sec: list             # [ [sector 0/1/2,...] per u ] geometric wedge
    # conflicts
    conflict_pairs: np.ndarray # (P,2) candidate index pairs
    # cross-hex pass-2 support
    fixed_open: np.ndarray     # (J,) bool: neighbor-owned sites forced open
    ext_residual: dict         # j -> [mhz]*3 residual per sector for those
    # NTN
    hex_ids: list              # hexes present (owned hex first)
    beam_cap_mbps: float

def build_instance(user_lat, user_lon, user_mbps,
                   hex_id: str,
                   tiers=DEFAULT_TIERS,
                   se_fn=se_default,
                   ext_sites=None,   # neighbor-owned OPEN sites: list of dicts
                                     # {lat, lon, tier_name, residual_mhz:[3]}
                                     # appear as cost-0 fixed-open candidates
                                     # with per-sector RESIDUAL capacity
                   rho_cand: float = 0.30,
                   rho_dep: float = 0.95,
                   agg_res: int = 9,
                   K_elig: int = 6,
                   se_min: float = 0.5,
                   beam_bw_hz: float = 300e6,
                   beam_se: float = 1.77,
                   halo_km: float = 2.847,
                   density_res: int = 9,
                   cross_tier_midband_conflict: bool = True) -> Instance:
    """Build one hex instance (+halo) from raw user positions.

    user_* : ALL users of hex_id plus its halo band (caller extracts them).
    """
    import h3
    user_lat = np.asarray(user_lat, dtype=np.float64)
    user_lon = np.asarray(user_lon, dtype=np.float64)
    user_mbps = np.asarray(user_mbps, dtype=np.float64)
    lat0 = float(user_lat.mean())
    ux, uy = project_km(user_lat, user_lon, lat0)

    # -- local density (people per km^2 at density_res) for tier zoning -----
    dcell = [h3.latlng_to_cell(float(a), float(b), density_res)
             for a, b in zip(user_lat, user_lon)]
    from collections import Counter
    dcount = Counter(dcell)
    cell_area = h3.average_hexagon_area(density_res, unit='km^2')
    dens = np.array([dcount[c] / cell_area for c in dcell])

    # -- candidates: per-tier lattice over populated points of eligible density
    cand_x, cand_y, cand_t, cand_c = [], [], [], []
    for ti, tier in enumerate(tiers):
        m = dens >= tier.density_min
        if m.sum() == 0:
            continue
        spacing = tier.radius_km * math.sqrt(3.0) * rho_cand
        nx, ny, cnt = tri_lattice(ux[m], uy[m], spacing)
        keep = cnt >= 1
        cand_x.append(nx[keep]); cand_y.append(ny[keep])
        cand_t.append(np.full(keep.sum(), ti))
        cand_c.append(np.full(keep.sum(), tier.cost))
    cand_xy = np.column_stack([np.concatenate(cand_x), np.concatenate(cand_y)])
    cand_tier = np.concatenate(cand_t).astype(np.int32)
    cand_cost = np.concatenate(cand_c)

    # append neighbor-owned external sites (fixed open, zero cost, residual W)
    tier_index = {t.name: i for i, t in enumerate(tiers)}
    n_own = len(cand_tier)
    ext_residual = {}
    if ext_sites:
        ex_x, ex_y = project_km(
            np.array([e["lat"] for e in ext_sites]),
            np.array([e["lon"] for e in ext_sites]), lat0)
        cand_xy = np.vstack([cand_xy, np.column_stack([ex_x, ex_y])])
        cand_tier = np.concatenate([cand_tier,
            np.array([tier_index[e["tier_name"]] for e in ext_sites],
                     dtype=np.int32)])
        cand_cost = np.concatenate([cand_cost, np.zeros(len(ext_sites))])
        for off, e in enumerate(ext_sites):
            ext_residual[n_own + off] = list(e["residual_mhz"])
    fixed_open = np.zeros(len(cand_tier), dtype=bool)
    fixed_open[n_own:] = True

    clat, clon = unproject(cand_xy[:, 0], cand_xy[:, 1], lat0)
    cand_owner = [h3.latlng_to_cell(float(a), float(b), 5)
                  for a, b in zip(clat, clon)]

    # -- demand aggregation to agg_res hex centroids -------------------------
    acell = [h3.latlng_to_cell(float(a), float(b), agg_res)
             for a, b in zip(user_lat, user_lon)]
    agg = {}
    for c, d in zip(acell, user_mbps):
        agg[c] = agg.get(c, 0.0) + float(d)
    dem_cells = list(agg.keys())
    dl = np.array([h3.cell_to_latlng(c) for c in dem_cells])
    dx_, dy_ = project_km(dl[:, 0], dl[:, 1], lat0)
    dem_xy = np.column_stack([dx_, dy_])
    dem_mbps = np.array([agg[c] for c in dem_cells])
    dem_hex = [h3.cell_to_parent(c, 5) for c in dem_cells]

    # -- eligibility: K best candidates (by SE) within radius ---------------
    # For each (demand point, candidate) pair we also record WHICH 120-degree
    # sector of the site serves the point (3GPP boresights 30/150/270 deg;
    # geometric wedge assignment by bearing -- deterministic, no decision).
    def _sector_of(cand_p, dem_p):
        dx = dem_p[0] - cand_p[0]      # east
        dy = dem_p[1] - cand_p[1]      # north
        bearing = math.degrees(math.atan2(dx, dy)) % 360.0
        # sector 0 boresight 30  -> wedge [330, 90)
        # sector 1 boresight 150 -> wedge [ 90,210)
        # sector 2 boresight 270 -> wedge [210,330)
        return int(((bearing - 330.0) % 360.0) // 120.0)

    tree = cKDTree(cand_xy)
    rmax = max(t.radius_km for t in tiers)
    elig_j, elig_se, elig_sec = [], [], []
    for i in range(len(dem_xy)):
        near = tree.query_ball_point(dem_xy[i], rmax)
        pairs = []
        for j in near:
            t = tiers[cand_tier[j]]
            d = float(np.hypot(*(cand_xy[j] - dem_xy[i])))
            if d > t.radius_km:
                continue
            se = se_fn(d, t)
            if se >= se_min:
                pairs.append((se, j, _sector_of(cand_xy[j], dem_xy[i])))
        pairs.sort(reverse=True)
        pairs = pairs[:K_elig]
        elig_j.append([j for _, j, _k in pairs])
        elig_se.append([se for se, _j, _k in pairs])
        elig_sec.append([k for _se, _j, k in pairs])

    # -- conflict pairs: same band, distance < d_min(tier) ------------------
    conflicts = []
    for band in ("mid", "mmw"):          # low band (RMa) never conflicts
        idx = np.array([j for j in range(len(cand_tier))
                        if band_of(tiers[cand_tier[j]]) == band])
        if len(idx) < 2:
            continue
        sub = cKDTree(cand_xy[idx])
        dmin_band = max(tiers[t].radius_km for t in set(cand_tier[idx])) \
            * math.sqrt(3.0) * rho_dep
        for a, b in sub.query_pairs(dmin_band):
            ja, jb = int(idx[a]), int(idx[b])
            ta, tb = tiers[cand_tier[ja]], tiers[cand_tier[jb]]
            same_tier = cand_tier[ja] == cand_tier[jb]
            if same_tier or cross_tier_midband_conflict:
                # use the smaller tier's d_min for the actual test
                dmin = min(ta.radius_km, tb.radius_km) * math.sqrt(3.0) * rho_dep
                if np.hypot(*(cand_xy[ja] - cand_xy[jb])) < dmin:
                    conflicts.append((ja, jb))
    conflict_pairs = np.array(sorted(set(conflicts)), dtype=np.int64) \
        if conflicts else np.zeros((0, 2), dtype=np.int64)

    hex_ids = [hex_id] + sorted({h for h in dem_hex if h != hex_id})
    return Instance(cand_xy=cand_xy, cand_tier=cand_tier, cand_cost=cand_cost,
                    cand_owner_hex=cand_owner, tiers=tuple(tiers), lat0=lat0,
                    dem_xy=dem_xy, dem_mbps=dem_mbps, dem_hex=dem_hex,
                    elig_j=elig_j, elig_se=elig_se, elig_sec=elig_sec,
                    conflict_pairs=conflict_pairs,
                    fixed_open=fixed_open, ext_residual=ext_residual,
                    hex_ids=hex_ids,
                    beam_cap_mbps=beam_bw_hz * beam_se / 1e6)
