"""
General-purpose utilities used across the hybrid NTN optimizer.

All functions are pure (no side effects) and unit-tested independently.
"""

from __future__ import annotations

import math
from typing import Tuple

from hybrid_ntn_optimizer.core.constants import EARTH_RADIUS_M, EARTH_MU
import os

def _detect_cpus() -> int:
    """CPUs actually allocated to this process — not the node's physical count.
    SLURM allocation first, then scheduler affinity (respects the cgroup cpuset),
    physical count only as a last resort."""
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n:
        return int(n)
    try:
        return len(os.sched_getaffinity(0))   # Linux: respects SLURM's cpuset
    except AttributeError:                     # macOS/Windows
        return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def wrap_degrees(angle_deg: float) -> float:
    """Wrap an angle to [0, 360)."""
    return angle_deg % 360.0


def wrap_degrees_signed(angle_deg: float) -> float:
    """Wrap an angle to (-180, +180]."""
    wrapped = angle_deg % 360.0
    return wrapped - 360.0 if wrapped > 180.0 else wrapped


def deg2rad(deg: float) -> float:
    return math.radians(deg)


def rad2deg(rad: float) -> float:
    return math.degrees(rad)


# ---------------------------------------------------------------------------
# Orbital mechanics helpers
# ---------------------------------------------------------------------------

def orbital_period_s(semi_major_axis_m: float) -> float:
    """
    Compute Keplerian orbital period (s) for a given semi-major axis.

    Uses the standard two-body formula:  T = 2π √(a³ / μ)

    Parameters
    ----------
    semi_major_axis_m : float
        Semi-major axis in metres.

    Returns
    -------
    float
        Orbital period in seconds.
    """
    return 2.0 * math.pi * math.sqrt(semi_major_axis_m**3 / EARTH_MU)


def orbital_velocity_m_s(semi_major_axis_m: float) -> float:
    """
    Mean circular orbital speed (m/s) at the given semi-major axis.
    v = √(μ / a)
    """
    return math.sqrt(EARTH_MU / semi_major_axis_m)


def altitude_to_sma(altitude_km: float) -> float:
    """Convert orbital altitude (km) to semi-major axis (m)."""
    return EARTH_RADIUS_M + altitude_km * 1_000.0


def sma_to_altitude_km(sma_m: float) -> float:
    """Convert semi-major axis (m) to orbital altitude (km)."""
    return (sma_m - EARTH_RADIUS_M) / 1_000.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_distance(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
    radius_m: float = EARTH_RADIUS_M,
) -> float:
    """
    Haversine great-circle distance between two geodetic points (metres).

    Parameters
    ----------
    lat1_deg, lon1_deg : float
        First point latitude and longitude (degrees).
    lat2_deg, lon2_deg : float
        Second point latitude and longitude (degrees).
    radius_m : float
        Sphere radius (default: Earth mean radius).

    Returns
    -------
    float
        Arc distance in metres.
    """
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    dlat = lat2 - lat1
    dlon = math.radians(lon2_deg - lon1_deg)

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return radius_m * 2.0 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Walker-Delta spacing helpers
# ---------------------------------------------------------------------------

def walker_raan_spacing_deg(num_planes: int) -> float:
    """RAAN spacing between adjacent planes in a Walker constellation."""
    return 360.0 / num_planes


def walker_phase_offset_deg(phasing: int, num_planes: int) -> float:
    """
    In-plane phase offset (degrees) between a satellite in plane k and
    plane k+1.  F is the phasing parameter (0 ≤ F < P).
    """
    return (phasing * 360.0) / (num_planes * (num_planes))


def mean_anomaly_spacing_deg(sats_per_plane: int) -> float:
    """Uniform mean-anomaly spacing for satellites within one plane."""
    return 360.0 / sats_per_plane


# ---------------------------------------------------------------------------
# ECI ↔ sub-satellite point
# ---------------------------------------------------------------------------

def eci_to_geodetic(
    x: float, y: float, z: float, gmst_rad: float = 0.0
) -> Tuple[float, float, float]:
    """
    Convert ECI position (m) to geodetic lat/lon (degrees) and altitude (m).

    Spherical-Earth approximation (no flattening).

    Parameters
    ----------
    x, y, z : float
        ECI position in metres.
    gmst_rad : float
        Greenwich Mean Sidereal Time in radians (for Earth-fixed frame).

    Returns
    -------
    (lat_deg, lon_deg, alt_m) : tuple[float, float, float]
    """
    r = math.sqrt(x**2 + y**2 + z**2)
    lat_rad = math.asin(z / r)
    lon_rad = math.atan2(y, x) - gmst_rad

    lat_deg = math.degrees(lat_rad)
    lon_deg = wrap_degrees_signed(math.degrees(lon_rad))
    alt_m = r - EARTH_RADIUS_M

    return lat_deg, lon_deg, alt_m

    """
    Approximate Greenwich Mean Sidereal Time (radians) from Julian Date.

    Uses the IAU 1982 model (good to ~0.1 s over decades).

    Parameters
    ----------
    epoch_jd : float
        Julian Date (e.g. 2451545.0 = J2000.0).

    Returns
    -------
    float
        GMST in radians, wrapped to [0, 2π).
    """
    T = (epoch_jd - 2_451_545.0) / 36_525.0
    theta_s = 67_310.548_41 + (8_640_184.812_866 + (0.093_104 - 6.2e-6 * T) * T) * T
    gmst_rad = math.radians(theta_s / 240.0) % (2.0 * math.pi)
    return gmst_rad