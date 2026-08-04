# milp/candidate_generator.py

#!/usr/bin/env python3
"""
candidate_generator.py
======================
Generates everything the per-hex MILP needs from raw user positions.

FIXES IN THIS REVISION (2026-07):
  * Instance now REMEMBERS the aggregation settings it was built with
    (agg_res, agg_safety, r_eff_km). refine.py needs them to shrink the
    fine-candidate radius the same way build_instance does -- previously the
    fine pass used the RAW tier radius, so refined candidates were credited
    with coverage the coarse pass had (correctly) refused them.
  * Guard against an empty candidate set (all density gates rejecting every
    point) -- np.concatenate([]) used to raise instead of saying why.
  * Guard against an empty demand set.
  * agg_res default raised 9 -> 10. Measured: res-10 (66 m circumradius)
    cut the model-vs-reality outage gap substantially versus res-9 (201 m),
    because eligibility is tested at the demand-point CENTROID and the
    circumradius is subtracted from every tier radius (see agg_safety).

All physics knobs are EXPLICIT parameters -- nothing hidden in defaults.
The SE function is pluggable so the cluster version calls the real
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
    cost: float                 # annualised TCO [USD/yr]
    density_min: float          # people/km^2 to be a candidate zone

DEFAULT_TIERS = (
    Tier("RMA",     700e6,  20e6, 2.847, 46.0, 15.0, 35.0, cost=38114.0, density_min=0.0),
    Tier("UMA",     3.5e9, 100e6, 0.474, 46.0, 17.0, 25.0, cost=13249.0, density_min=400.0),
    Tier("UMI",     3.5e9, 100e6, 0.311, 38.0, 10.0, 10.0, cost=2937.0, density_min=1000.0),
    Tier("UMI_MMW",  28e9, 400e6, 0.150, 30.0, 23.0, 10.0, cost=3714.0, density_min=7500.0),
)

def band_of(tier: Tier) -> str:
    if tier.freq_hz < 1e9:   return "low"
    if tier.freq_hz < 7e9:   return "mid"
    return "mmw"

# ---------------------------------------------------------------------------
# Pluggable spectral-efficiency model (replaced by real sinr.py on cluster)
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

def sector_of(cand_p, dem_p):
    """3GPP 3-sector boresights 30/150/270 deg; geometric wedge assignment by
    bearing -- deterministic, NOT a decision variable."""
    dx = dem_p[0] - cand_p[0]      # east
    dy = dem_p[1] - cand_p[1]      # north
    bearing = math.degrees(math.atan2(dx, dy)) % 360.0
    return int(((bearing - 330.0) % 360.0) // 120.0)

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
    beam_cap_of: dict          # hex_id -> usable beam capacity [Mbps]
    # cross-hex pass-2 support
    fixed_open: np.ndarray     # (J,) bool: neighbor-owned sites forced open
    ext_residual: dict         # j -> [mhz]*3 residual per sector
    # NTN
    hex_ids: list              # hexes present (owned hex first)
    beam_cap_mbps: float
    # ---- provenance of the aggregation (NEW: refine.py needs these) -------
    agg_res: int = 10
    agg_safety: float = 1.0
    r_eff_km: tuple = ()       # per-tier EFFECTIVE radius actually used
    dem_cell: tuple = ()       # aggregation-cell id per demand point
    dem_n: tuple = ()          # user count per demand point (risk sizing)
    dem_var: tuple = ()        # sum_u d_u^2 per cell = compound-Poisson var


def build_instance(user_lat, user_lon, user_mbps,
                   hex_id: str,
                   tiers=DEFAULT_TIERS,
                   se_fn=se_default,
                   beam_residual=None,  # {hex_id: Mbps} remaining beam capacity
                                        # for FOREIGN hexes (owned hex keeps full
                                        # C_beam). Prevents the same physical beam
                                        # being spent by several hex solves.
                   ext_sites=None,   # neighbor-owned OPEN sites: list of dicts
                                     # {lat, lon, tier_name, residual_mhz:[3]}
                   extra_cand=None,  # ADAPTIVE ENRICHMENT: [(lat,lon,tier_name)]
                   agg_safety: float = 1.0,  # DEMAND-AGGREGATION COVERAGE FIX.
                                     # Eligibility is tested at the res-`agg_res`
                                     # CENTROID, but a real user sits anywhere in
                                     # that hexagon -- up to its circumradius
                                     # away. Effective radius is shrunk by
                                     # agg_safety * circumradius:
                                     #   1.0 = every user in the demand point is
                                     #         guaranteed in range (conservative)
                                     #   0.0 = centroid only (optimistic)
                   rho_cand: float = 0.30,
                   rho_dep: float = 0.95,
                   agg_res: int = 10,
                   K_elig: int = 6,
                   se_min: float = 0.5,
                   beam_bw_hz: float = 300e6,
                   beam_se: float = 1.77,
                   halo_km: float = 2.847,
                   density_res: int = 9,
                   cross_tier_min_dist_km="legacy",
                                     # CROSS-TIER minimum separation.
                                     #   "legacy" (DEFAULT, unchanged behaviour)
                                     #        = min(R_a,R_b)*sqrt(3)*rho_dep
                                     #   float = explicit separation in km
                                     #   None  = cross-tier unconstrained
                                     #
                                     # DO NOT LOWER THIS FOR CO-CHANNEL TIERS.
                                     # UMa and UMi are both 3.5 GHz / 100 MHz
                                     # here, and UMa runs 46 dBm + 17 dBi vs
                                     # UMi 38 + 10 -- 15 dB more EIRP on the
                                     # SAME channel. A co-channel UMa overlay
                                     # costs UMi about 10 dB of SINR (SE 2.60
                                     # -> 0.89, -66% in the 3GPP UMi model);
                                     # measured on hex 852b9bd7, adding 392
                                     # UMa sites cut UMi SE 4.87 -> 3.50.
                                     # Real co-channel HetNets make the macro
                                     # +small overlay work with eICIC / ABS or
                                     # separate carriers -- neither of which
                                     # this simulator models. Lower it only if
                                     # the tiers are on DIFFERENT carriers.
                                     #
                                     # The knob exists so the assumption is
                                     # explicit and testable, not because the
                                     # default should change.
                   cross_tier_midband_conflict=None,   # DEPRECATED alias:
                                     # True  -> same as "legacy"
                                     # False -> same as None
                   ) -> Instance:
    """Build one hex instance (+halo) from raw user positions.

    user_* : ALL users of hex_id plus its halo band (caller extracts them).
    """
    import h3
    user_lat = np.asarray(user_lat, dtype=np.float64)
    user_lon = np.asarray(user_lon, dtype=np.float64)
    user_mbps = np.asarray(user_mbps, dtype=np.float64)
    if user_lat.size == 0:
        raise ValueError(f"build_instance({hex_id}): no users passed in")
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
        if keep.sum() == 0:
            continue
        cand_x.append(nx[keep]); cand_y.append(ny[keep])
        cand_t.append(np.full(keep.sum(), ti))
        cand_c.append(np.full(keep.sum(), tier.cost))
    if not cand_x:
        raise ValueError(
            f"build_instance({hex_id}): EVERY tier density gate rejected every "
            f"user (max local density {dens.max():.0f}/km^2 at res {density_res}"
            f" vs gates { {t.name: t.density_min for t in tiers} }). Lower the "
            f"gates or raise density_res.")
    cand_xy = np.column_stack([np.concatenate(cand_x), np.concatenate(cand_y)])
    cand_tier = np.concatenate(cand_t).astype(np.int32)
    cand_cost = np.concatenate(cand_c)

    # ---- adaptive enrichment: candidates the SOLVER asked for -------------
    if extra_cand:
        name_to_t = {t.name: i for i, t in enumerate(tiers)}
        ex, ey, et = [], [], []
        for (la, lo, tname) in extra_cand:
            if tname not in name_to_t:
                continue
            px, py = project_km(np.array([la]), np.array([lo]), lat0)
            ex.append(float(px[0])); ey.append(float(py[0]))
            et.append(name_to_t[tname])
        if ex:
            add_xy = np.column_stack([np.array(ex), np.array(ey)])
            keep = []
            for i in range(len(ex)):
                same = cand_tier == et[i]
                if same.any():
                    d = np.hypot(cand_xy[same, 0] - add_xy[i, 0],
                                 cand_xy[same, 1] - add_xy[i, 1])
                    if d.min() < 0.025:      # near-duplicate, <25 m
                        continue
                keep.append(i)
            if keep:
                add_xy = add_xy[keep]
                add_t = np.array([et[i] for i in keep], dtype=np.int32)
                cand_xy = np.vstack([cand_xy, add_xy])
                cand_tier = np.concatenate([cand_tier, add_t])
                cand_cost = np.concatenate(
                    [cand_cost, np.array([tiers[t].cost for t in add_t])])

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
    agg, agg_n, agg_d2 = {}, {}, {}
    for c, d in zip(acell, user_mbps):
        d = float(d)
        agg[c] = agg.get(c, 0.0) + d
        agg_n[c] = agg_n.get(c, 0) + 1
        agg_d2[c] = agg_d2.get(c, 0.0) + d * d
    dem_cells = list(agg.keys())
    if not dem_cells:
        raise ValueError(f"build_instance({hex_id}): no demand points")
    dl = np.array([h3.cell_to_latlng(c) for c in dem_cells])
    dx_, dy_ = project_km(dl[:, 0], dl[:, 1], lat0)
    dem_xy = np.column_stack([dx_, dy_])
    dem_mbps = np.array([agg[c] for c in dem_cells])
    dem_hex = [h3.cell_to_parent(c, 5) for c in dem_cells]

    # effective coverage radius per tier, corrected for demand aggregation
    _circ = h3.average_hexagon_edge_length(agg_res, unit="km") * agg_safety
    _r_eff = [max(t.radius_km - _circ, 0.02) for t in tiers]

    # -- eligibility: K best candidates (by SE) within EFFECTIVE radius -----
    tree = cKDTree(cand_xy)
    rmax = max(_r_eff)
    elig_j, elig_se, elig_sec = [], [], []
    for i in range(len(dem_xy)):
        near = tree.query_ball_point(dem_xy[i], rmax)
        pairs = []
        for j in near:
            t = tiers[cand_tier[j]]
            d = float(np.hypot(*(cand_xy[j] - dem_xy[i])))
            if d > _r_eff[cand_tier[j]]:
                continue
            se = se_fn(d, t)
            if se >= se_min:
                pairs.append((se, j, sector_of(cand_xy[j], dem_xy[i])))
        pairs.sort(reverse=True)
        pairs = pairs[:K_elig]
        elig_j.append([j for _, j, _k in pairs])
        elig_se.append([se for se, _j, _k in pairs])
        elig_sec.append([k for _se, _j, k in pairs])

    # -- conflict pairs -----------------------------------------------------
    # SAME TIER  : the design ISD. Co-tier cells of radius R tile the plane at
    #              ISD = R*sqrt(3); rho_dep is how tightly you may pack
    #              relative to that. This is EXACTLY the geometry se_fn assumed
    #              (6 interferers on a ring at R*sqrt(3)*rho_dep), so the
    #              constraint and the SE stay self-consistent.
    # CROSS TIER : a DIFFERENT question. Not "how do I tile the plane with
    #              co-tier cells" but "how close before the small cell is
    #              pointless". Answer is tens of metres, not hundreds. The
    #              legacy rule used the SMALL tier's full ISD and thereby
    #              forbade the macro+small-cell overlay real HetNets are built
    #              on -- see cross_tier_min_dist_km above.
    # LOW BAND (RMa) never conflicts with anything.
    _xt = cross_tier_min_dist_km
    if cross_tier_midband_conflict is not None:      # deprecated alias
        _xt = "legacy" if cross_tier_midband_conflict else None
    conflicts = []
    for band in ("mid", "mmw"):
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
            if cand_tier[ja] == cand_tier[jb]:
                dmin = ta.radius_km * math.sqrt(3.0) * rho_dep
            elif _xt is None:
                continue                              # cross-tier free
            elif _xt == "legacy":
                dmin = min(ta.radius_km, tb.radius_km) * math.sqrt(3.0) * rho_dep
            else:
                dmin = float(_xt)
            if np.hypot(*(cand_xy[ja] - cand_xy[jb])) < dmin:
                conflicts.append((ja, jb))
    # drop conflicts between TWO fixed-open external sites: both are already
    # built by neighbouring hexes, so their proximity is a fact, not a
    # decision -- pass 2 must not become infeasible over it.
    conflicts = [(a, b) for (a, b) in conflicts
                 if not (fixed_open[a] and fixed_open[b])]
    conflict_pairs = np.array(sorted(set(conflicts)), dtype=np.int64) \
        if conflicts else np.zeros((0, 2), dtype=np.int64)

    hex_ids = [hex_id] + sorted({h for h in dem_hex if h != hex_id})
    _cap = beam_bw_hz * beam_se / 1e6
    beam_cap_of = {h: (_cap if h == hex_id
                       else float((beam_residual or {}).get(h, _cap)))
                   for h in hex_ids}
    return Instance(cand_xy=cand_xy, cand_tier=cand_tier, cand_cost=cand_cost,
                    cand_owner_hex=cand_owner, tiers=tuple(tiers), lat0=lat0,
                    dem_xy=dem_xy, dem_mbps=dem_mbps, dem_hex=dem_hex,
                    elig_j=elig_j, elig_se=elig_se, elig_sec=elig_sec,
                    conflict_pairs=conflict_pairs, beam_cap_of=beam_cap_of,
                    fixed_open=fixed_open, ext_residual=ext_residual,
                    hex_ids=hex_ids, beam_cap_mbps=_cap,
                    agg_res=int(agg_res), agg_safety=float(agg_safety),
                    r_eff_km=tuple(_r_eff),
                    dem_cell=tuple(dem_cells),
                    dem_n=tuple(agg_n[c] for c in dem_cells),
                    dem_var=tuple(agg_d2[c] for c in dem_cells))