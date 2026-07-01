"""
Physical, orbital, and system-level constants.

All values are in SI units unless explicitly noted.
"""

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_M_S: float = 2.998_292_458e8      # m/s  (exact IAU value)
BOLTZMANN_CONSTANT: float = 1.380_649e-23         # J/K  (exact SI value)
EARTH_RADIUS_M: float = 6_371_000.0               # m    mean spherical radius
EARTH_MU: float = 3.986_004_418e14                # m³/s²  standard gravitational parameter
EARTH_J2: float = 1.082_626_68e-3                 # dimensionless  second zonal harmonic
EARTH_ROTATION_RAD_S: float = 7.292_115e-5        # rad/s  sidereal rotation rate

# ---------------------------------------------------------------------------
# Frequency bands (Hz) – centre frequencies for reference
# ---------------------------------------------------------------------------
FREQ_KU_HZ: float = 12.0e9    # Ku-band  (downlink reference, Starlink Gen-1)
FREQ_KA_HZ: float = 26.5e9    # Ka-band  (Starlink Gen-2 / V2)
FREQ_S_HZ: float  =  2.4e9    # S-band   (mobile NTN, NB-IoT)
FREQ_L_HZ: float  =  1.6e9    # L-band   (Iridium / Inmarsat reference)

# ---------------------------------------------------------------------------
# Orbit altitude bands (km) – used for classification
# ---------------------------------------------------------------------------
LEO_ALT_RANGE_KM = (200.0, 2_000.0)
MEO_ALT_RANGE_KM = (2_000.0, 35_786.0)
GEO_ALT_KM       = 35_786.0          # geostationary altitude
GEO_TOLERANCE_KM = 200.0             # ± km considered GEO

# ---------------------------------------------------------------------------
# Link-budget defaults (can be overridden via config)
# ---------------------------------------------------------------------------
NOISE_TEMPERATURE_K: float = 290.0        # K  (standard reference temperature)
ANTENNA_EFFICIENCY: float  = 0.55         # dimensionless
SYSTEM_NOISE_FIGURE_DB: float = 3.0       # dB  receiver noise figure
ATMOSPHERIC_LOSS_DB: float = 0.5          # dB  clear-sky margin (Ku-band, 10° el.)
RAIN_FADE_MARGIN_DB: float = 3.0          # dB  ITU-R P.618 temperate climate

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------
DEFAULT_TIME_STEP_S: float = 60.0         # s   propagation / snapshot cadence
DEFAULT_MIN_ELEVATION_DEG: float = 25.0   # deg minimum elevation angle for service
DEFAULT_EPOCH: str = "2024-01-01T00:00:00"  # ISO-8601 UTC



# Effective environment height h_E [m] for breakpoint computation
# TR 38.901 Table 7.4.1-1 Note 1: applies to both UMa and UMi
_H_E = 1.0   # [m]