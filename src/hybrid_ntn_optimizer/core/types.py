from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OrbitType(str, Enum):
    """Orbital regime classification."""
    LEO = "LEO"   # Low Earth Orbit  (200 – 2 000 km)
    MEO = "MEO"   # Medium Earth Orbit (2 000 – 35 786 km)
    GEO = "GEO"   # Geostationary Orbit (~35 786 km)
    HEO = "HEO"   # Highly Elliptical Orbit (e.g. Molniya)


class ConstellationType(str, Enum):
    """Pattern / geometry of the constellation."""
    WALKER_DELTA  = "walker_delta"   # inclined, used by Starlink / OneWeb
    WALKER_STAR   = "walker_star"    # polar / near-polar (e.g. Telesat Lightspeed)
    GEO_BELT      = "geo_belt"       # geostationary arc
    CUSTOM        = "custom"         # user-defined orbital elements


class FrequencyBand(str, Enum):
    """Spectrum band identifier."""
    L  = "L"
    S  = "S"
    C  = "C"
    X  = "X"
    KU = "Ku"
    KA = "Ka"
    V  = "V"
    Q  = "Q"


class PolarizationType(str, Enum):
    LHCP  = "LHCP"    # Left-Hand Circular
    RHCP  = "RHCP"    # Right-Hand Circular
    LINEAR_H = "H"
    LINEAR_V = "V"


class BeamShape(str, Enum):
    CIRCULAR  = "circular"
    ELLIPTICAL = "elliptical"
    HEXAGONAL = "hexagonal"   # spot-beam footprint approximation


# ---------------------------------------------------------------------------
# Coordinate / position types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeoPoint:
    """Geodetic coordinate (WGS-84 ellipsoid approximated as sphere)."""
    lat_deg: float   # –90 … +90
    lon_deg: float   # –180 … +180

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat_deg <= 90.0):
            raise ValueError(f"lat_deg must be in [-90, 90], got {self.lat_deg}")
        if not (-180.0 <= self.lon_deg <= 180.0):
            raise ValueError(f"lon_deg must be in [-180, 180], got {self.lon_deg}")

    @property
    def lat_rad(self) -> float:
        return math.radians(self.lat_deg)

    @property
    def lon_rad(self) -> float:
        return math.radians(self.lon_deg)


@dataclass(frozen=True)
class ECIVector:
    """Earth-Centred Inertial position/velocity vector (metres or m/s)."""
    x: float
    y: float
    z: float

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __add__(self, other: ECIVector) -> ECIVector:
        return ECIVector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: ECIVector) -> ECIVector:
        return ECIVector(self.x - other.x, self.y - other.y, self.z - other.z)


# ---------------------------------------------------------------------------
# Orbital element types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeplerianElements:
    """
    Classical (two-body) Keplerian orbital elements.

    Parameters
    ----------
    semi_major_axis_m : float
        Semi-major axis in metres.
    eccentricity : float
        Orbital eccentricity (0 = circular).
    inclination_deg : float
        Inclination in degrees (0 = equatorial, 90 = polar).
    raan_deg : float
        Right Ascension of the Ascending Node in degrees.
    arg_perigee_deg : float
        Argument of perigee in degrees.
    true_anomaly_deg : float
        True anomaly at epoch in degrees.
    """
    semi_major_axis_m: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    true_anomaly_deg: float

    def __post_init__(self) -> None:
        if self.eccentricity < 0.0 or self.eccentricity >= 1.0:
            raise ValueError(
                f"eccentricity must be in [0, 1), got {self.eccentricity}"
            )
        if self.semi_major_axis_m <= 0.0:
            raise ValueError("semi_major_axis_m must be positive")


# ---------------------------------------------------------------------------
# Constellation / satellite descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkerParameters:
    """
    Walker-Delta (T/P/F) constellation descriptor.

    T   : total number of satellites
    P   : number of orbital planes
    F   : phasing parameter (relative spacing between planes)

    The inclination and altitude are the same for all planes in a
    Walker-Delta shell.
    """
    total_satellites: int       # T
    num_planes: int             # P
    phasing: int                # F  (0 ≤ F < P)
    inclination_deg: float
    altitude_km: float
    orbit_type: OrbitType = OrbitType.LEO
    constellation_type: ConstellationType = ConstellationType.WALKER_DELTA

    @property
    def sats_per_plane(self) -> int:
        if self.total_satellites % self.num_planes != 0:
            raise ValueError(
                f"total_satellites ({self.total_satellites}) must be "
                f"divisible by num_planes ({self.num_planes})"
            )
        return self.total_satellites // self.num_planes

    @property
    def altitude_m(self) -> float:
        return self.altitude_km * 1_000.0

# ---------------------------------------------------------------------------
# Visibility / link result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisibilityRecord:
    """
    Result of a satellite-to-ground visibility check at one time instant.
    """
    satellite_id: str
    ground_lat_deg: float
    ground_lon_deg: float
    epoch_utc: str
    elevation_deg: float      # degrees above horizon
    azimuth_deg: float        # degrees from North, clockwise
    slant_range_m: float      # straight-line distance in metres
    is_visible: bool          # True if elevation ≥ min_elevation threshold

    @property
    def slant_range_km(self) -> float:
        return self.slant_range_m / 1_000.0