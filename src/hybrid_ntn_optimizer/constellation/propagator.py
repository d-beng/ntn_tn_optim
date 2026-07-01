"""
Orbital propagator backed by Skyfield + SGP4.

Replaces the manual Keplerian + J2 implementation with a proper SGP4
propagator.  The public API is identical to the previous version so
nothing above this layer needs to change.

Architecture
------------
Walker-Delta generates ``KeplerianElements`` (osculating, t=0).
This module converts them to SGP4 mean elements, wraps each satellite
in a Skyfield ``EarthSatellite``, and delegates all propagation and
coordinate transforms to Skyfield.

Why SGP4 over raw Keplerian?
-----------------------------
* Handles atmospheric drag (via B* term — set to zero for clean
  theoretical orbits, or realistic values for fidelity).
* Full SGP4/SDP4 perturbation model (J2, J3, J4, lunar-solar for HEO).
* Skyfield's coordinate pipeline is numerically correct: ECI → ECEF →
  geodetic uses proper Earth rotation (GMST/ERA), polar motion, etc.
* ``EarthSatellite`` objects are reusable across many time steps with
  no recomputation of static quantities.

Limitations
-----------
* SGP4 is a *mean*-element theory.  The ``KeplerianElements`` from
  ``walker_delta.py`` are osculating elements.  For circular orbits at
  LEO altitudes the difference is negligible (< 1 km) for coverage
  analysis.  For high-accuracy ranging, convert osculating → mean first
  (e.g. via the Brouwer transform).
* Epoch: all satellites are initialised at the same reference epoch.
  Relative phasing is encoded in the initial mean anomaly, exactly as
  in the parametric Walker-Delta geometry.
"""

from __future__ import annotations

import math
from typing import List

from sgp4.api import Satrec, WGS84
from skyfield.api import EarthSatellite, load, wgs84

from hybrid_ntn_optimizer.core.constants import (
    DEFAULT_EPOCH,
    DEFAULT_TIME_STEP_S,
    EARTH_MU,
)
from datetime import datetime, timedelta, timezone
from hybrid_ntn_optimizer.core.exceptions import PropagationError
from hybrid_ntn_optimizer.core.types import (
    ECIVector,
    KeplerianElements,
)

from hybrid_ntn_optimizer.models.satellite import SatelliteDescriptor, SatelliteState


# Module-level Skyfield timescale — expensive to create, share one instance
_TS = load.timescale()


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def iso8601_to_jd(epoch_utc: str) -> float:
    """Convert an ISO-8601 UTC string (``YYYY-MM-DDTHH:MM:SS``) to Julian Date."""
    try:
        date_part, time_part = epoch_utc.split("T")
        y, mo, d = (int(x) for x in date_part.split("-"))
        h, m, s  = (float(x) for x in time_part.split(":"))
    except (ValueError, AttributeError) as exc:
        raise PropagationError(
            f"Cannot parse epoch string {epoch_utc!r}. "
            "Expected format: YYYY-MM-DDTHH:MM:SS"
        ) from exc

    if mo <= 2:
        y -= 1
        mo += 12
    A  = int(y / 100)
    B  = 2 - A + int(A / 4)
    jd = (
        int(365.25 * (y + 4716))
        + int(30.6001 * (mo + 1))
        + d + B - 1524.5
        + (h + m / 60.0 + s / 3600.0) / 24.0
    )
    return jd

"""
def advance_epoch(epoch_utc: str, dt_s: float) -> str:
    jd_new = iso8601_to_jd(epoch_utc) + dt_s / 86_400.0
    t = _TS.tt_jd(jd_new)
    utc = t.utc
    return (
        f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}"
        f"T{utc.hour:02d}:{utc.minute:02d}:{int(utc.second):02d}"
    )
"""

def advance_epoch(epoch_utc: str, dt_s: float) -> str:
    """
    Advance an ISO-8601 UTC epoch by ``dt_s`` seconds using standard datetime.
    Returns a new ISO-8601 string (UTC, second precision).
    """
    # Parse the string into a datetime object
    dt_obj = datetime.strptime(epoch_utc, "%Y-%m-%dT%H:%M:%S")
    
    # Add the seconds accurately
    new_dt = dt_obj + timedelta(seconds=dt_s)
    
    # Format back to string
    return new_dt.strftime("%Y-%m-%dT%H:%M:%S")

# ---------------------------------------------------------------------------
# KeplerianElements → Skyfield EarthSatellite
# ---------------------------------------------------------------------------

def _mean_motion_rad_min(semi_major_axis_m: float) -> float:
    """SGP4 mean motion in radians per minute."""
    return math.sqrt(EARTH_MU / semi_major_axis_m**3) * 60.0


def build_earth_satellite(
    descriptor: SatelliteDescriptor,
    epoch_utc: str,
    bstar: float = 0.0,
) -> EarthSatellite:
    """
    Convert a ``SatelliteDescriptor`` (Walker-Delta output) into a Skyfield
    ``EarthSatellite`` ready for propagation.

    Parameters
    ----------
    descriptor : SatelliteDescriptor
        Satellite with initial Keplerian elements.
    epoch_utc : str
        Reference epoch (ISO-8601 UTC).
    bstar : float
        SGP4 drag term (m⁻¹).  0.0 = drag-free theoretical constellation.

    Returns
    -------
    EarthSatellite
    """
    el  = descriptor.elements
    jd  = iso8601_to_jd(epoch_utc)
    # SGP4 epoch: days from 1949-12-31 00:00 UT  (JD 2433281.5)
    sgp4_epoch = jd - 2_433_281.5

    sat = Satrec()
    sat.sgp4init(
        WGS84,
        "i",                                        # opsmode ('i' = improved)
        hash(descriptor.sat_id) % 100_000,          # satnum
        sgp4_epoch,
        bstar,
        0.0,                                        # ndot
        0.0,                                        # nddot
        el.eccentricity,
        math.radians(el.arg_perigee_deg),           # argpo (rad)
        math.radians(el.inclination_deg),           # inclo (rad)
        math.radians(el.true_anomaly_deg),          # mo — mean anomaly (rad)
        _mean_motion_rad_min(el.semi_major_axis_m),
        math.radians(el.raan_deg),                  # nodeo — RAAN (rad)
    )
    return EarthSatellite.from_satrec(sat, _TS)


# ---------------------------------------------------------------------------
# Core propagation functions
# ---------------------------------------------------------------------------

def propagate_satellite(
    descriptor: SatelliteDescriptor,
    epoch_utc: str,
    dt_s: float,
    apply_j2: bool = True,
    _sat_cache: dict | None = None,
) -> SatelliteState:
    """
    Propagate one satellite to ``dt_s`` seconds after ``epoch_utc``.

    Parameters
    ----------
    descriptor : SatelliteDescriptor
    epoch_utc : str
    dt_s : float
        Elapsed time in seconds.
    apply_j2 : bool
        Accepted for API compatibility. SGP4 always includes J2.
    _sat_cache : dict, optional
        Pass a shared dict to avoid rebuilding the Satrec on every call.

    Returns
    -------
    SatelliteState
    """
    if _sat_cache is not None and descriptor.sat_id in _sat_cache:
        earth_sat = _sat_cache[descriptor.sat_id]
    else:
        earth_sat = build_earth_satellite(descriptor, epoch_utc)
        if _sat_cache is not None:
            _sat_cache[descriptor.sat_id] = earth_sat
    """
    jd_target = iso8601_to_jd(epoch_utc) + dt_s / 86_400.0
    t = _TS.tt_jd(jd_target)
    """
    target_epoch = advance_epoch(epoch_utc, dt_s)
    dt_obj = datetime.strptime(target_epoch, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    t = _TS.from_datetime(dt_obj)
    geocentric = earth_sat.at(t)
    pos_m  = geocentric.position.m
    vel_ms = geocentric.velocity.m_per_s
    lat_lon = wgs84.latlon_of(geocentric)
    lat_deg = float(lat_lon[0].degrees)
    lon_deg = float(lat_lon[1].degrees)
    alt_m   = float(wgs84.height_of(geocentric).m)

    return SatelliteState(
        satellite_id=descriptor.sat_id,
        epoch_utc=advance_epoch(epoch_utc, dt_s),
        position_eci=ECIVector(float(pos_m[0]),  float(pos_m[1]),  float(pos_m[2])),
        velocity_eci=ECIVector(float(vel_ms[0]), float(vel_ms[1]), float(vel_ms[2])),
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        altitude_m=alt_m,
    )


def propagate_constellation(
    descriptors: List[SatelliteDescriptor],
    epoch_utc: str,
    dt_s: float,
    apply_j2: bool = True,
) -> List[SatelliteState]:
    """
    Propagate all satellites to a single epoch.

    Builds all ``EarthSatellite`` objects once then evaluates them all
    at the same Skyfield ``Time`` — no redundant work.

    Parameters
    ----------
    descriptors : list[SatelliteDescriptor]
    epoch_utc : str
    dt_s : float
    apply_j2 : bool
        Accepted for API compatibility.

    Returns
    -------
    list[SatelliteState]
        Same order as input.
    """
    earth_sats  = [build_earth_satellite(d, epoch_utc) for d in descriptors]
    """
    jd_target   = iso8601_to_jd(epoch_utc) + dt_s / 86_400.0
    t           = _TS.tt_jd(jd_target)
    target_epoch = advance_epoch(epoch_utc, dt_s)
    """
    target_epoch = advance_epoch(epoch_utc, dt_s)
    dt_obj = datetime.strptime(target_epoch, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    t = _TS.from_datetime(dt_obj)
    states: List[SatelliteState] = []
    for desc, earth_sat in zip(descriptors, earth_sats):
        geocentric = earth_sat.at(t)
        pos_m  = geocentric.position.m
        vel_ms = geocentric.velocity.m_per_s
        lat_lon = wgs84.latlon_of(geocentric)
        lat_deg = float(lat_lon[0].degrees)
        lon_deg = float(lat_lon[1].degrees)
        alt_m   = float(wgs84.height_of(geocentric).m)

        states.append(SatelliteState(
            satellite_id=desc.sat_id,
            epoch_utc=target_epoch,
            position_eci=ECIVector(float(pos_m[0]),  float(pos_m[1]),  float(pos_m[2])),
            velocity_eci=ECIVector(float(vel_ms[0]), float(vel_ms[1]), float(vel_ms[2])),
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            altitude_m=alt_m,
        ))

    return states


def generate_ground_track(
    descriptor: SatelliteDescriptor,
    epoch_utc: str,
    duration_s: float,
    time_step_s: float = DEFAULT_TIME_STEP_S,
    apply_j2: bool = True,
) -> List[SatelliteState]:
    """
    Generate a time series of ``SatelliteState`` for one satellite.

    Builds the ``EarthSatellite`` once and propagates to all time steps
    in a single vectorised Skyfield call — much faster than a Python loop.

    Parameters
    ----------
    descriptor : SatelliteDescriptor
    epoch_utc : str
    duration_s : float
    time_step_s : float
    apply_j2 : bool
        Accepted for API compatibility.

    Returns
    -------
    list[SatelliteState]
    """
    earth_sat = build_earth_satellite(descriptor, epoch_utc)

    times_s = [i * time_step_s for i in range(int(duration_s / time_step_s) + 1)]
    base_dt = datetime.strptime(epoch_utc, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

    # Build a list of accurate UTC datetime objects
    dt_list = [base_dt + timedelta(seconds=dt) for dt in times_s]

    # Tell Skyfield to parse the list of datetimes directly
    t_arr = _TS.from_datetimes(dt_list)

    geocentric = earth_sat.at(t_arr)
    pos_m  = geocentric.position.m        # shape (3, N)
    vel_ms = geocentric.velocity.m_per_s  # shape (3, N)
    lats, lons = wgs84.latlon_of(geocentric)
    alts   = wgs84.height_of(geocentric).m

    states: List[SatelliteState] = []
    for i, dt in enumerate(times_s):
        # latlon_of returns arrays when given an array of times
        lat = lats[i].degrees if hasattr(lats, "__getitem__") else lats.degrees
        lon = lons[i].degrees if hasattr(lons, "__getitem__") else lons.degrees
        alt = float(alts[i])  if hasattr(alts, "__getitem__") else float(alts)

        states.append(SatelliteState(
            satellite_id=descriptor.sat_id,
            epoch_utc=advance_epoch(epoch_utc, dt),
            position_eci=ECIVector(float(pos_m[0, i]), float(pos_m[1, i]), float(pos_m[2, i])),
            velocity_eci=ECIVector(float(vel_ms[0, i]), float(vel_ms[1, i]), float(vel_ms[2, i])),
            lat_deg=float(lat),
            lon_deg=float(lon),
            altitude_m=alt,
        ))

    return states