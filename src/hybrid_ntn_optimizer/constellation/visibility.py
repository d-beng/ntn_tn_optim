"""
Satellite visibility — elevation, azimuth, slant range, and coverage.

Uses Skyfield's topocentric geometry for all angle/range computations.
This is more accurate than the manual spherical-Earth formulas it replaces:
Skyfield accounts for the WGS-84 ellipsoid, proper Earth rotation (ERA),
and aberration.

Public API  (unchanged from the previous version)
--------------------------------------------------
``check_visibility``               single sat × single ground point × one epoch
``visible_satellites``             all sats visible from one point at one epoch
``best_satellite``                 highest-elevation satellite
``coverage_snapshot``              per-grid-cell metrics at one epoch
``coverage_fraction``              scalar coverage fraction from a cell list
``instantaneous_coverage_radius_km``  geometric upper bound (no propagation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

from skyfield.api import EarthSatellite, load, wgs84

from hybrid_ntn_optimizer.core.constants import DEFAULT_MIN_ELEVATION_DEG
from hybrid_ntn_optimizer.core.types import (
    GeoPoint,
    VisibilityRecord,
)

from hybrid_ntn_optimizer.models.satellite import SatelliteDescriptor, SatelliteState

from hybrid_ntn_optimizer.constellation.propagator import (
    _TS,
    build_earth_satellite,
    iso8601_to_jd,
)

# ---------------------------------------------------------------------------
# Internal helper — build a Skyfield observer from a GeoPoint
# ---------------------------------------------------------------------------

def _observer(ground: GeoPoint):
    """Return a Skyfield ``GeographicPosition`` for the given ground point."""
    return wgs84.latlon(ground.lat_deg, ground.lon_deg)


# ---------------------------------------------------------------------------
# Single visibility check
# ---------------------------------------------------------------------------

def check_visibility(
    state: SatelliteState,
    ground: GeoPoint,
    min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG,
    _earth_sat: EarthSatellite | None = None,
) -> VisibilityRecord:
    """
    Check whether a satellite is visible from a ground point.

    Parameters
    ----------
    state : SatelliteState
        Propagated satellite state.  ``epoch_utc`` is used as the time.
    ground : GeoPoint
        Observer's geodetic location.
    min_elevation_deg : float
        Minimum elevation angle for the link to be considered valid.
    _earth_sat : EarthSatellite, optional
        Pre-built Skyfield object.  Pass it to avoid rebuilding when
        calling in a tight loop (e.g. from ``visible_satellites``).

    Returns
    -------
    VisibilityRecord
    """
    # We need the EarthSatellite to compute topocentric angles.
    # If not supplied, reconstruct it from the descriptor embedded in state.
    # For bulk calls, callers should pass _earth_sat directly.
    if _earth_sat is None:
        raise ValueError(
            "check_visibility requires a pre-built EarthSatellite. "
            "Use visible_satellites() or coverage_snapshot() for bulk queries, "
            "or supply _earth_sat explicitly."
        )

    observer = _observer(ground)
    dt_obj = datetime.strptime(state.epoch_utc, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    t = _TS.from_datetime(dt_obj)
    difference  = _earth_sat - observer
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()

    el_deg = float(alt.degrees)
    az_deg = float(az.degrees)
    sr_m   = float(distance.m)

    return VisibilityRecord(
        satellite_id=state.satellite_id,
        ground_lat_deg=ground.lat_deg,
        ground_lon_deg=ground.lon_deg,
        epoch_utc=state.epoch_utc,
        elevation_deg=el_deg,
        azimuth_deg=az_deg,
        slant_range_m=sr_m,
        is_visible=el_deg >= min_elevation_deg,
    )


# ---------------------------------------------------------------------------
# Multi-satellite visible set
# ---------------------------------------------------------------------------

def visible_satellites(
    states: List[SatelliteState],
    ground: GeoPoint,
    min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG,
    earth_sats: List[EarthSatellite] | None = None,
) -> List[VisibilityRecord]:
    """
    Return visibility records for all visible satellites from ``ground``.

    Results are sorted by descending elevation (best link first).

    Parameters
    ----------
    states : list[SatelliteState]
        Constellation snapshot (all at the same epoch).
    ground : GeoPoint
    min_elevation_deg : float
    earth_sats : list[EarthSatellite], optional
        Pre-built Skyfield objects, same order as ``states``.
        If None, they are built from state data (requires descriptors
        to be accessible — prefer passing them explicitly).

    Returns
    -------
    list[VisibilityRecord]
        Only ``is_visible=True`` records, sorted by elevation ↓.
    """
    if earth_sats is None:
        raise ValueError(
            "visible_satellites requires pre-built EarthSatellite objects. "
            "Use LEOConstellation.visible_from() which handles this automatically."
        )

    observer = _observer(ground)
    dt_obj = datetime.strptime(states[0].epoch_utc, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    t = _TS.from_datetime(dt_obj)
    records: List[VisibilityRecord] = []
    for state, earth_sat in zip(states, earth_sats):
        diff  = earth_sat - observer
        topo  = diff.at(t)
        alt, az, dist = topo.altaz()

        el_deg = float(alt.degrees)
        if el_deg < min_elevation_deg:
            continue

        records.append(VisibilityRecord(
            satellite_id=state.satellite_id,
            ground_lat_deg=ground.lat_deg,
            ground_lon_deg=ground.lon_deg,
            epoch_utc=state.epoch_utc,
            elevation_deg=el_deg,
            azimuth_deg=float(az.degrees),
            slant_range_m=float(dist.m),
            is_visible=True,
        ))

    return sorted(records, key=lambda r: r.elevation_deg, reverse=True)


def best_satellite(
    states: List[SatelliteState],
    ground: GeoPoint,
    min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG,
    earth_sats: List[EarthSatellite] | None = None,
) -> Optional[VisibilityRecord]:
    """Highest-elevation visible satellite, or None."""
    vis = visible_satellites(states, ground, min_elevation_deg, earth_sats)
    return vis[0] if vis else None


# ---------------------------------------------------------------------------
# Grid-level coverage snapshot
# ---------------------------------------------------------------------------

@dataclass
class CoverageCell:
    """Coverage metrics for one geographic grid cell at one epoch."""
    lat_deg: float
    lon_deg: float
    num_visible: int = 0
    best_elevation_deg: float = -90.0
    best_slant_range_m: float = float("inf")
    best_satellite_id: Optional[str] = None
    is_covered: bool = False


def coverage_snapshot(
    states: List[SatelliteState],
    lat_grid: List[float],
    lon_grid: List[float],
    min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG,
    earth_sats: List[EarthSatellite] | None = None,
) -> List[CoverageCell]:
    """
    Compute per-cell coverage metrics for a geographic grid at one epoch.

    Parameters
    ----------
    states : list[SatelliteState]
        All satellite states at a single epoch.
    lat_grid : list[float]
    lon_grid : list[float]
    min_elevation_deg : float
    earth_sats : list[EarthSatellite], optional
        Pre-built Skyfield objects (same order as ``states``).

    Returns
    -------
    list[CoverageCell]
        Row-major order (lat outer, lon inner).
    """
    if earth_sats is None:
        raise ValueError(
            "coverage_snapshot requires pre-built EarthSatellite objects. "
            "Use LEOConstellation.coverage_at() which handles this automatically."
        )

    dt_obj = datetime.strptime(states[0].epoch_utc, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    t = _TS.from_datetime(dt_obj)
    cells: List[CoverageCell] = []
    for lat in lat_grid:
        for lon in lon_grid:
            observer = wgs84.latlon(lat, lon)
            cell = CoverageCell(lat_deg=lat, lon_deg=lon)
            best_el  = -90.0
            best_sr  = float("inf")
            best_id  = None
            count    = 0

            for state, earth_sat in zip(states, earth_sats):
                diff = earth_sat - observer
                topo = diff.at(t)
                alt, az, dist = topo.altaz()
                el = float(alt.degrees)
                if el >= min_elevation_deg:
                    count += 1
                    if el > best_el:
                        best_el = el
                        best_sr = float(dist.m)
                        best_id = state.satellite_id

            cell.num_visible       = count
            cell.is_covered        = count > 0
            cell.best_elevation_deg = best_el if count > 0 else -90.0
            cell.best_slant_range_m = best_sr if count > 0 else float("inf")
            cell.best_satellite_id  = best_id
            cells.append(cell)

    return cells


def coverage_fraction(cells: List[CoverageCell]) -> float:
    """
    Fraction of grid cells with at least one satellite in view.

    Returns
    -------
    float
        Coverage fraction in [0, 1].
    """
    if not cells:
        return 0.0
    return sum(1 for c in cells if c.is_covered) / len(cells)


# ---------------------------------------------------------------------------
# Theoretical maximum coverage radius (geometry only, no Skyfield needed)
# ---------------------------------------------------------------------------

def instantaneous_coverage_radius_km(
    altitude_km: float,
    min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG,
) -> float:
    """
    Radius of the instantaneous coverage circle on the Earth's surface (km).

    Uses the spherical-Earth central-angle formula.  This is a geometric
    upper bound — Skyfield is not involved.

    Parameters
    ----------
    altitude_km : float
    min_elevation_deg : float

    Returns
    -------
    float
        Coverage circle radius in km.
    """
    from hybrid_ntn_optimizer.core.constants import EARTH_RADIUS_M
    re_km = EARTH_RADIUS_M / 1_000.0
    rs_km = re_km + altitude_km
    el_rad = math.radians(min_elevation_deg)
    rho_rad = math.acos((re_km / rs_km) * math.cos(el_rad)) - el_rad
    return re_km * rho_rad