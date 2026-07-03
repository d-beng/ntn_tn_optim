"""
3GPP sector antenna geometry + horizontal pattern.

References:
  - 3GPP TR 38.901 (system-level calibration: 3-sector macro sites, sector
    boresight defines the GCS x-axis).
  - 3GPP TR 36.942 v15 clause 4.2.1 (horizontal antenna pattern):
        A(theta) = -min[ 12 * (theta / theta_3dB)^2 , A_m ]  [dB]
    with theta_3dB = 65 deg and A_m = 30 dB for a 3-sector macro site.

This module is pure geometry/dB; it does not change any of your RF numbers.
Import it from full_pipeline.py and sinr.py.
"""
import math

THETA_3DB_DEG = 65.0     # 3GPP TR 36.942 3-sector half-power beamwidth
A_MAX_DB = 30.0          # 3GPP TR 36.942 front-to-back / max attenuation


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing (deg, 0=North, clockwise) from point1 -> point2."""
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute angular difference between two bearings (0..180)."""
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d


def sector_gain_db(bs_lat, bs_lon, sector_azimuth_deg, ue_lat, ue_lon):
    """Horizontal sector antenna gain offset (dB, <=0) toward a UE.

    Returns 0.0 for an omni cell (sector_azimuth_deg is None).
    For a sectored cell, returns the TR 36.942 pattern value A(theta), where
    theta is the angle between the sector boresight and the BS->UE direction.
    This is an OFFSET added to the cell's g_tx_dbi (boresight gain)."""
    if sector_azimuth_deg is None:
        return 0.0
    brg = bearing_deg(bs_lat, bs_lon, ue_lat, ue_lon)
    theta = angle_diff_deg(brg, sector_azimuth_deg)
    return -min(12.0 * (theta / THETA_3DB_DEG) ** 2, A_MAX_DB)


def in_sector(bs_lat, bs_lon, sector_azimuth_deg, ue_lat, ue_lon, half_width_deg=60.0):
    """True if the UE lies within this sector's angular wedge (+/-half_width).
    Omni cells (azimuth None) always return True. Default half-width 60 deg
    gives three non-overlapping 120-deg wedges for a 3-sector site."""
    if sector_azimuth_deg is None:
        return True
    brg = bearing_deg(bs_lat, bs_lon, ue_lat, ue_lon)
    return angle_diff_deg(brg, sector_azimuth_deg) <= half_width_deg