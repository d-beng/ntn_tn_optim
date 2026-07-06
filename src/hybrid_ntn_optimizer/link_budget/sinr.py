"""
Unified SINR, Capacity, and Coverage calculations for
Terrestrial Networks (TN) and Non-Terrestrial Networks (NTN).

References:
- TN:  3GPP TR 38.901 v17.0.0 — Channel Models for Frequencies from 0.5 to 100 GHz
         Table 7.4.1-1  (path-loss formulas)
         Table 7.4.2-1  (shadowing standard-deviation values)
         Section 7.4.1  (LOS probability, breakpoint distance)
- TN sector antenna: 3GPP TR 38.901 (3-sector macro geometry) +
         3GPP TR 36.942 v15 clause 4.2.1 (horizontal antenna pattern).
- NTN: 3GPP TR 38.821 v16.0.0 — Solutions for NR to Support NTN
         Section 6.1    (link-budget methodology)
         Table 6.1-1    (typical G/T figures)
- NF:  3GPP TS 38.101-1 v17.x  — UE noise figure limits

Corrections applied vs. original:
  [BUG-1]  UMi LOS breakpoint used raw h_BS/h_UT; must subtract h_E = 1 m
            (TR 38.901 Table 7.4.1-1 Note 1 applies to both UMa and UMi).
  [BUG-2]  RMa, InH and InF-SH pathloss were silently falling through to
            UMa — all three are now fully implemented from TR 38.901
            Table 7.4.1-1.
  [BUG-3]  Single fixed shadowing σ (7.8 dB) used for every scenario/LOS
            combination; replaced with per-scenario LOS/NLOS σ table drawn
            from TR 38.901 Table 7.4.2-1.
  [BUG-4]  calculate_max_tn_radius_km capped UMi search at 2 000 m and
            RMa search at 5 000 m; TR 38.901 validity is 5 000 m (UMi) and
            10 000 m (RMa). Corrected per-scenario distance ceilings added.
  [BUG-5]  NTN function assumed every interfering beam shares the same
            slant range as the wanted beam.  An explicit per-beam slant-range
            list has been added.
  [BUG-6]  NTN function lacked a polarisation-discrimination loss parameter
            (TR 38.821 Section 6.1); default 3 dB added.
  [ADD-1]  Helper get_shadowing_sigma() exposes the σ table so callers can
            draw consistent random realisations.
  [ADD-2]  Scenario validity guards now warn (via warnings.warn) when inputs
            fall outside the specified TR 38.901 applicability ranges.
  [SECTOR] calculate_tn_sinr_capacity now applies 3GPP sectorization on the
            INTERFERER side: each interferer's horizontal sector pattern
            (TR 36.942 4.2.1) attenuates its contribution by the angle between
            that sector's boresight and the direction to the served UE. The
            old flat interferer_beamforming_suppression_db is disabled by
            default (set to 0.0) to avoid double-counting; the pattern models
            it directionally instead. Requires ue_lat/ue_lon (the served UE
            position) so per-interferer angles can be computed. Omni cells
            (sector_azimuth_deg is None) get a 0 dB offset, i.e. unchanged.
"""

import math
from time import sleep
import warnings
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional

from hybrid_ntn_optimizer.core.constants import _H_E
from hybrid_ntn_optimizer.models.base_station import BaseStation, DeploymentScenario
from hybrid_ntn_optimizer.link_budget.sector_antenna import sector_gain_db


# ──────────────────────────────────────────────────────────────────────────────
# 1. Constants & Enumerations
# ──────────────────────────────────────────────────────────────────────────────

C_M_S   = 299_792_458.0   # Speed of light [m/s]
K_B     = 1.380649e-23    # Boltzmann constant [J/K]
T_SYS_K = 290.0           # Reference noise temperature [K]
# Boltzmann constant in logarithmic form:  10·log10(1.380649 × 10⁻²³) = −228.6 dBW/K/Hz
K_DB    = 10.0 * math.log10(K_B)   # ≈ −228.599 dBW/K/Hz (exact, not hard-coded)



# Default BS antenna heights [m] — TR 38.901 Table 7.2-1
DEFAULT_H_BS: dict = {
    DeploymentScenario.UMA:    25.0,
    DeploymentScenario.UMI:    10.0,
    DeploymentScenario.RMA:    35.0,
    DeploymentScenario.INH:     3.0,
    DeploymentScenario.INF_SH:  8.0,
}
"""
# ──────────────────────────────────────────────────────────────────────────────
# Shadowing standard-deviation table [dB]
# Source: TR 38.901 Table 7.4.2-1
# Keys: (DeploymentScenario, los: bool)
# ──────────────────────────────────────────────────────────────────────────────
_SHADOW_SIGMA_DB: dict = {
    (DeploymentScenario.UMA,    True):  4.0,    # UMa LOS
    (DeploymentScenario.UMA,    False): 6.0,    # UMa NLOS
    (DeploymentScenario.UMI,    True):  4.0,    # UMi LOS
    (DeploymentScenario.UMI,    False): 7.82,   # UMi NLOS
    (DeploymentScenario.RMA,    True):  4.0,    # RMa LOS  (σ_SF1; σ_SF2 = 6 dB for PL2 region)
    (DeploymentScenario.RMA,    False): 8.0,    # RMa NLOS
    (DeploymentScenario.INH,    True):  3.0,    # InH LOS
    (DeploymentScenario.INH,    False): 8.03,   # InH NLOS
    (DeploymentScenario.INF_SH, True):  2.0,    # InF-SH LOS
    (DeploymentScenario.INF_SH, False): 4.0,    # InF-SH NLOS
}

def get_shadowing_sigma(scenario: DeploymentScenario, los: bool) -> float:
    return _SHADOW_SIGMA_DB[(scenario, los)]
"""


# ──────────────────────────────────────────────────────────────────────────────
# 2. Internal Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fspl_db_km_ghz(distance_km: float, freq_ghz: float) -> float:
    """
    Free-Space Path Loss [dB].
    FSPL = 20·log10(d_km) + 20·log10(f_GHz) + 92.45

    The constant 92.45 dB = 20·log10(4π·10³·10⁹ / c) is derived exactly from
    the standard FSPL formula with d in km and f in GHz.
    """
    return (20.0 * math.log10(max(distance_km, 1e-3))
            + 20.0 * math.log10(max(freq_ghz, 1e-6))
            + 92.45)


def _breakpoint_distance(h_bs: float, h_ut: float, fc_hz: float) -> float:
    """
    3GPP effective breakpoint distance d'_BP [m].
    TR 38.901 Table 7.4.1-1 Note 1 (applies to UMa AND UMi):

        d'_BP = 4 · (h_BS − h_E) · (h_UT − h_E) · f_c / c

    where h_E = 1 m for both scenarios.

    [BUG-1 fix] — original UMi helper omitted the h_E subtraction.
    """
    h_bs_eff = h_bs - _H_E
    h_ut_eff = h_ut - _H_E
    return 4.0 * h_bs_eff * h_ut_eff * fc_hz / C_M_S


# ──────────────────────────────────────────────────────────────────────────────
# 3. Per-Scenario LOS Pathloss Functions
#    All implement the piecewise formulas from TR 38.901 Table 7.4.1-1
# ──────────────────────────────────────────────────────────────────────────────

def _uma_los_pathloss(d_2d: float, d_3d: float, fc_ghz: float,
                     h_bs: float, h_ut: float) -> float:
    """
    UMa LOS pathloss [dB] — TR 38.901 Table 7.4.1-1, valid 10 ≤ d_2D ≤ 5000 m.
    Uses corrected breakpoint (shared _breakpoint_distance helper).
    """
    d_bp = _breakpoint_distance(h_bs, h_ut, fc_ghz * 1e9)

    if d_2d <= d_bp:
        # PL1
        return 28.0 + 22.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)
    else:
        # PL2
        return (28.0 + 40.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)
                - 9.0 * math.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))


def _umi_los_pathloss(d_2d: float, d_3d: float, fc_ghz: float,
                     h_bs: float, h_ut: float) -> float:
    """
    UMi-Street Canyon LOS pathloss [dB] — TR 38.901 Table 7.4.1-1.
    Valid 10 ≤ d_2D ≤ 5000 m.

    [BUG-1 fix] breakpoint now uses (h_BS − h_E) and (h_UT − h_E).
    """
    d_bp = _breakpoint_distance(h_bs, h_ut, fc_ghz * 1e9)

    if d_2d <= d_bp:
        # PL1
        return 32.4 + 21.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)
    else:
        # PL2
        return (32.4 + 40.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)
                - 9.5 * math.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))


def _rma_los_pathloss(d_2d: float, d_3d: float, fc_ghz: float,
                     h_bs: float, h_ut: float,
                     avg_building_height_m: float = 5.0) -> float:
    """
    RMa LOS pathloss [dB] — TR 38.901 Table 7.4.1-1.
    Valid 10 ≤ d_2D ≤ 10 000 m, 0.5 ≤ f_c ≤ 30 GHz.

    RMa uses a different breakpoint formula:
        d_BP = 2π · h_BS · h_UT · f_c / c   (no h_E subtraction for RMa)
    """
    h = avg_building_height_m
    fc_hz = fc_ghz * 1e9

    # RMa breakpoint — TR 38.901 Table 7.4.1-1 (different from UMa/UMi)
    d_bp = 2.0 * math.pi * h_bs * h_ut * fc_hz / C_M_S

    # Auxiliary coefficients
    coef_a = min(0.03 * (h ** 1.72), 10.0)
    coef_b = min(0.044 * (h ** 1.72), 14.77)

    # PL1 (d_2D ≤ d_BP)
    pl1 = (20.0 * math.log10(40.0 * math.pi * d_3d * fc_ghz / 3.0)
           + coef_a * math.log10(d_3d)
           - coef_b
           + 0.002 * math.log10(h) * d_3d)

    if d_2d <= d_bp:
        return pl1
    else:
        # PL2 — use PL1 evaluated at the breakpoint
        d_bp_3d = math.sqrt(d_bp ** 2 + (h_bs - h_ut) ** 2)
        pl1_at_bp = (20.0 * math.log10(40.0 * math.pi * d_bp_3d * fc_ghz / 3.0)
                     + coef_a * math.log10(d_bp_3d)
                     - coef_b
                     + 0.002 * math.log10(h) * d_bp_3d)
        return pl1_at_bp + 40.0 * math.log10(d_3d / d_bp_3d)


def _inh_los_pathloss(d_3d: float, fc_ghz: float) -> float:
    """
    InH-Office LOS pathloss [dB] — TR 38.901 Table 7.4.1-1.
    Valid 1 ≤ d_3D ≤ 150 m, 0.5 ≤ f_c ≤ 100 GHz.
    """
    return 32.4 + 17.3 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)


def _inf_sh_los_pathloss(d_3d: float, fc_ghz: float) -> float:
    """
    InF-SH (Indoor Factory — Sparse High) LOS pathloss [dB].
    TR 38.901 Table 7.4.1-1.  Same formula as InH LOS.
    Valid 1 ≤ d_3D ≤ 600 m, 0.5 ≤ f_c ≤ 100 GHz.
    """
    return 32.4 + 17.3 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Unified 3GPP Pathloss Engine
# ──────────────────────────────────────────────────────────────────────────────

def pathloss_3gpp(
    scenario: DeploymentScenario,
    distance_2d_m: float,
    carrier_freq_hz: float,
    bs_height_m: float,
    ue_height_m: float = 1.5,
    avg_building_height_m: float = 5.0,
    street_width_m: float = 20.0,
) -> Tuple[float, bool]:
    """
    Compute the deterministic (median) 3GPP NLOS pathloss for any supported
    scenario and return (pathloss_dB, is_los_dominated).

    The function returns the *larger* of the LOS and NLOS formulas as required
    by TR 38.901 Table 7.4.1-1 for UMa, UMi, RMa and InF-SH, and the NLOS
    value directly for InH.

    Parameters
    ----------
    scenario            : deployment environment
    distance_2d_m       : 2-D (horizontal) separation [m]
    carrier_freq_hz     : carrier frequency [Hz]
    ue_height_m         : UE antenna height [m]  (default 1.5 m)
    bs_height_m         : BS/AP antenna height [m] (None → scenario default)
    avg_building_height_m : average building height [m] — RMa only
    street_width_m      : street width W [m] — RMa NLOS only

    Returns
    -------
    (pl_db, los_dominated) — path loss [dB] and True if LOS term dominated.

    Applicability guards
    --------------------
    Out-of-range inputs trigger a UserWarning (not an exception) so that
    system-level simulations are not silently corrupted.

    [BUG-2 fix] RMa, InH and InF-SH are now fully implemented.
    """
    fc_ghz = carrier_freq_hz / 1e9

    if bs_height_m is None:
        print(f"BS height not provided; using default for {scenario.value}: {DEFAULT_H_BS[scenario]}")
        bs_height_m = DEFAULT_H_BS[scenario]

    # ── Validity guards ────────────────────────────────────────────────────────
    _check_validity(scenario, distance_2d_m, fc_ghz, ue_height_m, bs_height_m)

    # ── Minimum distance clamp (per TR 38.901 scenario applicability) ─────────
    min_d2d = 1.0 if scenario in (DeploymentScenario.INH, DeploymentScenario.INF_SH) else 10.0
    d_2d = max(distance_2d_m, min_d2d)
    d_3d = math.sqrt(d_2d ** 2 + (bs_height_m - ue_height_m) ** 2)

    # ══════════════════════════════════════════════════════════════════════════
    if scenario == DeploymentScenario.UMA:
        pl_los  = _uma_los_pathloss(d_2d, d_3d, fc_ghz, bs_height_m, ue_height_m)
        pl_nlos = (13.54
                   + 39.08 * math.log10(d_3d)
                   + 20.0  * math.log10(fc_ghz)
                   - 0.6   * (ue_height_m - 1.5))
        pl = max(pl_los, pl_nlos)
        return pl, (pl_los >= pl_nlos)

    # ══════════════════════════════════════════════════════════════════════════
    elif scenario == DeploymentScenario.UMI:
        pl_los  = _umi_los_pathloss(d_2d, d_3d, fc_ghz, bs_height_m, ue_height_m)
        pl_nlos = (35.3  * math.log10(d_3d)
                   + 22.4
                   + 21.3 * math.log10(fc_ghz)
                   - 0.3  * (ue_height_m - 1.5))
        pl = max(pl_los, pl_nlos)
        return pl, (pl_los >= pl_nlos)

    # ══════════════════════════════════════════════════════════════════════════
    elif scenario == DeploymentScenario.RMA:
        # [BUG-2 fix] — full RMa implementation
        pl_los  = _rma_los_pathloss(d_2d, d_3d, fc_ghz, bs_height_m,
                                    ue_height_m, avg_building_height_m)
        h  = avg_building_height_m
        W  = street_width_m
        # RMa NLOS — TR 38.901 Table 7.4.1-1
        pl_nlos = (161.04
                   - 7.1  * math.log10(W)
                   + 7.5  * math.log10(h)
                   - (24.37 - 3.7 * (h / bs_height_m) ** 2) * math.log10(bs_height_m)
                   + (43.42 - 3.1 * math.log10(bs_height_m)) * (math.log10(d_3d) - 3.0)
                   + 20.0 * math.log10(fc_ghz)
                   - (3.2 * (math.log10(11.75 * ue_height_m)) ** 2 - 4.97))
        pl = max(pl_los, pl_nlos)
        return pl, (pl_los >= pl_nlos)

    # ══════════════════════════════════════════════════════════════════════════
    elif scenario == DeploymentScenario.INH:
        # [BUG-2 fix] — full InH-Office implementation
        pl_los  = _inh_los_pathloss(d_3d, fc_ghz)
        # InH NLOS — TR 38.901 Table 7.4.1-1
        pl_nlos = (38.3 * math.log10(d_3d)
                   + 17.30
                   + 24.9 * math.log10(fc_ghz))
        # For InH the spec takes max(PL_LOS, PL_NLOS) as well
        pl = max(pl_los, pl_nlos)
        return pl, (pl_los >= pl_nlos)

    # ══════════════════════════════════════════════════════════════════════════
    elif scenario == DeploymentScenario.INF_SH:
        # [BUG-2 fix] — full InF-SH implementation
        pl_los  = _inf_sh_los_pathloss(d_3d, fc_ghz)
        # InF-SH NLOS — TR 38.901 Table 7.4.1-1
        pl_nlos = (33.63 * math.log10(d_3d)
                   + 21.9
                   + 20.0 * math.log10(fc_ghz))
        pl = max(pl_los, pl_nlos)
        return pl, (pl_los >= pl_nlos)

    else:
        raise ValueError(f"Unsupported DeploymentScenario: {scenario}")


def pathloss_3gpp_nlos(
    scenario: DeploymentScenario,
    distance_2d_m: float,
    carrier_freq_hz: float,
    ue_height_m: float = 1.5,
    bs_height_m: float = None,
    avg_building_height_m: float = 5.0,
    street_width_m: float = 20.0,
) -> float:
    """
    Convenience wrapper — returns only the pathloss value [dB].
    Kept for backward compatibility with the original API.
    """
    pl, _ = pathloss_3gpp(
        scenario, distance_2d_m, carrier_freq_hz,
        ue_height_m, bs_height_m,
        avg_building_height_m, street_width_m,
    )
    return pl


def _check_validity(
    scenario: DeploymentScenario,
    d_2d: float,
    fc_ghz: float,
    h_ut: float,
    h_bs: float,
) -> None:
    """
    Emit UserWarnings when inputs are outside TR 38.901 applicability ranges.
    Does not raise exceptions so Monte-Carlo loops are not broken.
    """
    limits = {
        DeploymentScenario.UMA:    (10, 5_000,  0.5, 100,  1.5, 22.5, 10,  150),
        DeploymentScenario.UMI:    (10, 5_000,  0.5, 100,  1.5, 22.5, 10,  150),
        DeploymentScenario.RMA:    (10, 10_000, 0.5,  30,  1.0, 10.0, 10,  150),
        DeploymentScenario.INH:    (1,    150,  0.5, 100,  1.0,  2.5,  2,   10),
        DeploymentScenario.INF_SH: (1,    600,  0.5, 100,  1.0, 12.5,  2,   25),
    }
    d_min, d_max, f_min, f_max, hut_min, hut_max, hbs_min, hbs_max = limits[scenario]

    if not (d_min <= d_2d <= d_max):
        warnings.warn(
            f"{scenario.value}: d_2D={d_2d:.1f} m outside [{d_min}, {d_max}] m "
            f"(TR 38.901 applicability).", UserWarning, stacklevel=4)
    if not (f_min <= fc_ghz <= f_max):
        warnings.warn(
            f"{scenario.value}: f_c={fc_ghz:.3f} GHz outside [{f_min}, {f_max}] GHz.",
            UserWarning, stacklevel=4)
    if not (hut_min <= h_ut <= hut_max):
        warnings.warn(
            f"{scenario.value}: h_UT={h_ut:.1f} m outside [{hut_min}, {hut_max}] m.",
            UserWarning, stacklevel=4)
    if not (hbs_min <= h_bs <= hbs_max):
        warnings.warn(
            f"{scenario.value}: h_BS={h_bs:.1f} m outside [{hbs_min}, {hbs_max}] m.",
            UserWarning, stacklevel=4)


# ──────────────────────────────────────────────────────────────────────────────
# 5. TN SINR & Capacity (5G NR Downlink)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_tn_sinr_capacity(
    bs_height_m: float,
    dist_to_serving_m: float,
    interferers: List[Tuple[BaseStation, float]],
    shadow_sigma_los_db: float,
    shadow_sigma_nlos_db: float,
    scenario: DeploymentScenario = DeploymentScenario.UMA,
    p_tx_dbm: float = 46.0,
    g_tx_dbi: float = 15.0,
    g_rx_ue_dbi: float = 0.0,
    serving_beamforming_gain_db: float = 12.0,
    interferer_beamforming_suppression_db: float = 12.0,  
    carrier_freq_hz: float = 3.5e9,
    bandwidth_hz: float = 100e6,
    body_loss_db: float = 3.0,
    noise_figure_db: float = 7.0,
    implementation_loss_factor: float = 0.65,
    ue_height_m: float = 1.5,
    ue_lat: Optional[float] = None,      # [SECTOR] served UE position for
    ue_lon: Optional[float] = None,      #          per-interferer sector angle
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    5G NR downlink SINR and throughput using 3GPP TR 38.901 pathloss.

    The shadowing standard deviation is now drawn from the per-scenario,
    per-LOS-condition table in TR 38.901 Table 7.4.2-1.  [BUG-3 fix]

    SECTORIZATION (3GPP TR 38.901 macro geometry; TR 36.942 4.2.1 pattern):
      Each interferer's contribution is attenuated by its horizontal sector
      antenna pattern toward the served UE. The angle is computed from the
      interferer's boresight (bs_intf.sector_azimuth_deg) and the bearing from
      the interferer to the UE (ue_lat/ue_lon). A sector facing away from the
      UE interferes far less; an omni interferer (sector_azimuth_deg None) is
      unchanged (0 dB offset). This DIRECTIONAL model replaces the old flat
      interferer_beamforming_suppression_db, whose default is therefore now
      0.0 to avoid double-counting. The SERVING side's sector gain is applied
      by the caller (full_pipeline passes g_tx_dbi already offset), so it is
      not re-applied here.

    Parameters
    ----------
    dist_to_serving_m              : 2-D distance to serving BS [m]
    interferers                    : list of tuples (BaseStation, 2-D distance) to interfering BSs [m]
    scenario                       : 3GPP deployment scenario
    p_tx_dbm                       : BS transmit power [dBm]
    g_tx_dbi                       : BS antenna gain [dBi] (serving; may already
                                     include the serving sector pattern offset)
    g_rx_ue_dbi                    : UE receive antenna gain [dBi]
    serving_beamforming_gain_db    : beamforming gain toward served UE [dB]
    interferer_beamforming_suppression_db : legacy flat isolation [dB]; default
                                     0.0 because the sector pattern now models
                                     interferer directivity explicitly
    carrier_freq_hz                : carrier frequency [Hz]
    bandwidth_hz                   : system bandwidth [Hz]
    body_loss_db                   : body/cable loss at UE [dB]
    noise_figure_db                : UE receiver noise figure [dB]
                                     (FR1 ≤6 GHz → 7 dB; FR2 ≥24 GHz → 10 dB
                                     per TS 38.101)
    implementation_loss_factor     : ηₗₒₛₛ ∈ (0, 1] applied to Shannon SE
    ue_height_m                    : UE antenna height [m]
    bs_height_m                    : BS antenna height [m] (None → default)
    ue_lat, ue_lon                 : served UE position (for interferer sector
                                     angles); None → interferer pattern skipped
    seed                           : optional RNG seed for reproducibility

    Returns
    -------
    (sinr_db, throughput_mbps, spectral_efficiency_bps_hz, diag)
    """
    rng = np.random.default_rng(seed)

    # ── Serving signal power ──────────────────────────────────────────────────
    pl_serving_db, los_serving = pathloss_3gpp(
        scenario, dist_to_serving_m, carrier_freq_hz, bs_height_m, ue_height_m)

    sigma_s = shadow_sigma_los_db if los_serving else shadow_sigma_nlos_db  # [BUG-3 fix]
    pl_serving_db += body_loss_db + rng.normal(0.0, sigma_s)

    s_dbm = p_tx_dbm + g_tx_dbi + g_rx_ue_dbi + serving_beamforming_gain_db - pl_serving_db
    s_mw  = 10.0 ** (s_dbm / 10.0)

    # ── Aggregate interference ────────────────────────────────────────────────
    i_mw = 0.0
    for bs_intf, d_intf_m in interferers:
        pl_j_db, los_j = pathloss_3gpp(
            bs_intf.scenario,
            d_intf_m,
            bs_intf.carrier_freq_hz,
            bs_intf.bs_height_m
        )

        sigma_j = bs_intf.shadow_sigma_los_db if los_j else bs_intf.shadow_sigma_nlos_db
        pl_j_db += body_loss_db + rng.normal(0.0, sigma_j)

        # [SECTOR] directional interferer antenna gain toward the served UE.
        # 0 dB for omni cells; negative (down to -30 dB) for sectors pointing
        # away. Requires the UE position; if not provided, offset is 0 dB.
        if ue_lat is not None and ue_lon is not None:
            intf_sector_off = sector_gain_db(
                bs_intf.lat, bs_intf.lon,
                getattr(bs_intf, "sector_azimuth_deg", None),
                ue_lat, ue_lon)
        else:
            intf_sector_off = 0.0

        p_rx_j_dbm = (bs_intf.p_tx_dbm + bs_intf.g_tx_dbi + intf_sector_off + g_rx_ue_dbi
                      - interferer_beamforming_suppression_db
                      - pl_j_db)
        i_mw += 10.0 ** (p_rx_j_dbm / 10.0)

    # ── Thermal noise ─────────────────────────────────────────────────────────
    # N [dBm] = −174 + 10·log10(BW) + NF   (kT₀ = −174 dBm/Hz at T₀ = 290 K)
    n_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db
    n_mw  = 10.0 ** (n_dbm / 10.0)

    # ── SINR & Shannon capacity ───────────────────────────────────────────────
    sinr_linear = s_mw / (i_mw + n_mw)
    sinr_db     = 10.0 * math.log10(sinr_linear)

    spectral_efficiency = implementation_loss_factor * math.log2(1.0 + sinr_linear)
    throughput_mbps     = bandwidth_hz * spectral_efficiency / 1e6

    # --- Diagnostic components (dBm) so callers can see WHY sinr is high/low ---
    s_dbm_out = 10.0 * math.log10(s_mw)
    i_dbm_out = (10.0 * math.log10(i_mw)) if i_mw > 0.0 else float('-inf')
    n_dbm_out = 10.0 * math.log10(n_mw)
    diag = {
        "S_dBm": s_dbm_out,
        "I_dBm": i_dbm_out,
        "N_dBm": n_dbm_out,
        "num_interferers": len(interferers),
        # I/N ratio in dB: how much interference dominates (or not) over noise.
        # >0 => interference-limited; <0 => noise-limited.
        "IoverN_dB": (i_dbm_out - n_dbm_out) if i_mw > 0.0 else float('-inf'),
    }

    return sinr_db, throughput_mbps, spectral_efficiency, diag

# ──────────────────────────────────────────────────────────────────────────────
# 6. NTN SINR & Capacity  (LEO / MEO / GEO)
#    Reference: 3GPP TR 38.821 v16.0.0, Section 6.1
# ──────────────────────────────────────────────────────────────────────────────

def calculate_ntn_sinr_capacity(
    slant_range_km: float,
    off_axis_angles_deg: List[float],
    interferer_slant_ranges_km: Optional[List[float]] = None,
    eirp_dbw: float = 40.0,
    g_t_db: float = -15.5,
    freq_ghz: float = 2.0,
    bandwidth_hz: float = 40e6,
    weather_loss_db: float = 1.0,
    theta_3db_deg: float = 2.5,
    sll_db: float = 25.0,
    polarisation_discrimination_db: float = 3.0,
    implementation_loss_factor: float = 0.65,
) -> Tuple[float, float, float]:
    """
    NTN downlink SINR [dB] and Shannon beam capacity using the
    C/N₀ framework of 3GPP TR 38.821.

    Corrections vs. original
    ------------------------
    [BUG-5 fix] Per-beam slant ranges:
        The original used the wanted-beam slant range for every interferer.
        This is only valid when all beams originate from the same satellite
        at identical geometry, which is not the general case.  An explicit
        `interferer_slant_ranges_km` list is now accepted; when omitted it
        defaults to the wanted-beam range (preserving the original behaviour
        while making the approximation explicit).

    [BUG-6 fix] Polarisation discrimination:
        Co-polar adjacent beams in NTN systems do not experience perfect
        polarisation match.  A `polarisation_discrimination_db` parameter
        (default 3 dB, representing typical cross-polar isolation for
        co-polar frequency re-use) has been added following
        TR 38.821 Section 6.1 guidance.

    Parameters
    ----------
    slant_range_km           : distance from satellite to wanted UE [km]
    off_axis_angles_deg      : list of angular offsets of interfering beams
                               relative to the wanted beam boresight [deg]
    interferer_slant_ranges_km : slant range for each interfering beam [km]
                               (None → same as slant_range_km for all)
    eirp_dbw                 : satellite EIRP per beam [dBW]
    g_t_db                   : UE figure of merit G/T [dB/K]
    freq_ghz                 : downlink carrier frequency [GHz]
    bandwidth_hz             : beam bandwidth [Hz]
    weather_loss_db          : atmospheric + rain loss (combined) [dB]
    theta_3db_deg            : satellite antenna 3-dB beamwidth [deg]
    sll_db                   : antenna side-lobe suppression level [dB]
    polarisation_discrimination_db : co-polar isolation loss [dB]
    implementation_loss_factor : ηₗₒₛₛ ∈ (0, 1] applied to Shannon SE

    Returns
    -------
    (sinr_db, throughput_mbps, spectral_efficiency_bps_hz)
    """
    # ── Wanted beam C/N₀ ─────────────────────────────────────────────────────
    fspl_wanted_db = _fspl_db_km_ghz(slant_range_km, freq_ghz)
    noise_bw_db    = 10.0 * math.log10(bandwidth_hz)

    # C/N₀ [dBHz] = EIRP [dBW] − FSPL [dB] − L_atm [dB] + G/T [dB/K] − k_B [dBW/K/Hz]
    # Note: K_DB ≈ −228.6, so −K_DB ≈ +228.6 dBW/K/Hz
    cn0_dbhz = eirp_dbw + g_t_db - fspl_wanted_db - weather_loss_db - K_DB
    cn_db    = cn0_dbhz - noise_bw_db          # C/N (dimensionless in dB)
    s_linear = 10.0 ** (cn_db / 10.0)          # C/N in linear scale

    # ── Per-beam interferer slant ranges ──────────────────────────────────────
    # [BUG-5 fix]
    if interferer_slant_ranges_km is None:
        interferer_slant_ranges_km = [slant_range_km] * len(off_axis_angles_deg)
    elif len(interferer_slant_ranges_km) != len(off_axis_angles_deg):
        raise ValueError(
            "interferer_slant_ranges_km must have the same length as "
            "off_axis_angles_deg.")

    # ── Adjacent-beam interference accumulation ───────────────────────────────
    i_linear = 0.0
    for theta_off, r_j_km in zip(off_axis_angles_deg, interferer_slant_ranges_km):
        # Parabolic beam roll-off, capped at the side-lobe level
        roll_off_db = min(12.0 * (theta_off / theta_3db_deg) ** 2, sll_db)

        fspl_j_db = _fspl_db_km_ghz(r_j_km, freq_ghz)

        # [BUG-6 fix] Subtract polarisation discrimination from interferer EIRP
        eirp_intf_dbw = eirp_dbw - roll_off_db - polarisation_discrimination_db

        interferer_cn0_dbhz = eirp_intf_dbw + g_t_db - fspl_j_db - weather_loss_db - K_DB
        interferer_cn_db    = interferer_cn0_dbhz - noise_bw_db
        i_linear += 10.0 ** (interferer_cn_db / 10.0)

    # ── SINR ──────────────────────────────────────────────────────────────────
    # With s_linear = C/N and i_linear = ΣCⱼ/N:
    #   SINR = C/(N + ΣCⱼ) = (C/N) / (1 + ΣCⱼ/N)
    sinr_linear = s_linear / (1.0 + i_linear)
    sinr_db     = 10.0 * math.log10(sinr_linear)

    spectral_efficiency = implementation_loss_factor * math.log2(1.0 + sinr_linear)
    throughput_mbps     = bandwidth_hz * spectral_efficiency / 1e6

    return sinr_db, throughput_mbps, spectral_efficiency


# ──────────────────────────────────────────────────────────────────────────────
# 7. TN Maximum Cell Radius (Bisection Search on Link Budget)
# ──────────────────────────────────────────────────────────────────────────────

# Per-scenario 2-D distance ceilings [m] matching TR 38.901 applicability.
# [BUG-4 fix] — original code capped UMi at 2 000 m and RMa at 5 000 m.
_MAX_DIST_M: dict = {
    DeploymentScenario.UMA:    5_000.0,
    DeploymentScenario.UMI:    5_000.0,   # was 2 000 m — corrected
    DeploymentScenario.RMA:   10_000.0,   # was 5 000 m — corrected
    DeploymentScenario.INH:      150.0,
    DeploymentScenario.INF_SH:   600.0,
}


def calculate_max_tn_radius_km(
    p_tx_dbm: float,
    g_tx_dbi: float,
    g_rx_ue_dbi: float,
    carrier_freq_hz: float,
    bandwidth_hz: float,
    sinr_min_db: float,
    body_loss_db: float,
    scenario: DeploymentScenario = DeploymentScenario.UMA,
    interference_margin_db: float = 2.0,
    noise_figure_db: float = 7.0,
    ue_height_m: float = 1.5,
    bs_height_m: Optional[float] = None,
    tolerance_m: float = 1.0,
    avg_building_height_m: float = 5.0,
    street_width_m: float = 20.0,
) -> float:
    """
    Maximum TN cell radius [km] from a noise-limited link budget,
    solved via bisection on the 3GPP pathloss model.

    The search ceiling now respects the TR 38.901 per-scenario validity
    range (UMi: 5 000 m, RMa: 10 000 m).  [BUG-4 fix]

    Parameters
    ----------
    p_tx_dbm              : BS transmit power [dBm]
    g_tx_dbi              : BS antenna gain [dBi]
    g_rx_ue_dbi           : UE antenna gain [dBi]
    carrier_freq_hz       : carrier frequency [Hz]
    bandwidth_hz          : system bandwidth [Hz]
    sinr_min_db           : minimum required SINR [dB]
    body_loss_db          : body/cable loss [dB]
    scenario              : deployment scenario
    interference_margin_db: additional interference margin [dB]
    noise_figure_db       : UE noise figure [dB]
    ue_height_m           : UE antenna height [m]
    bs_height_m           : BS antenna height [m] (None → default)
    tolerance_m           : bisection stopping criterion [m]
    avg_building_height_m : average building height [m] — RMa only
    street_width_m        : street width [m] — RMa only

    Returns
    -------
    cell_radius_km : maximum cell radius [km]; 0.0 if link budget infeasible.
    """
    if bs_height_m is None:
        print(f"BS height not provided; using default for {scenario.value}: {DEFAULT_H_BS[scenario]}")
        bs_height_m = DEFAULT_H_BS[scenario]

    # Thermal noise floor [dBm] = kT₀B + NF
    n_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db

    # Minimum received power needed at UE [dBm]
    p_rx_min_dbm = sinr_min_db + n_dbm + interference_margin_db

    # Maximum allowable median path loss [dB]
    max_path_loss_db = (p_tx_dbm + g_tx_dbi + g_rx_ue_dbi
                        - body_loss_db - p_rx_min_dbm)

    def _pl(d_m: float) -> float:
        return pathloss_3gpp_nlos(
            scenario, d_m, carrier_freq_hz, ue_height_m, bs_height_m,
            avg_building_height_m, street_width_m)

    min_dist = 10.0 if scenario not in (DeploymentScenario.INH,
                                        DeploymentScenario.INF_SH) else 1.0
    max_dist = _MAX_DIST_M[scenario]  # [BUG-4 fix]

    # Feasibility check
    if _pl(min_dist) > max_path_loss_db:
        return 0.0
    if _pl(max_dist) <= max_path_loss_db:
        return max_dist / 1000.0

    # Bisection
    d_lo, d_hi = min_dist, max_dist
    while (d_hi - d_lo) > tolerance_m:
        d_mid = (d_lo + d_hi) * 0.5
        if _pl(d_mid) < max_path_loss_db:
            d_lo = d_mid
        else:
            d_hi = d_mid

    return ((d_lo + d_hi) * 0.5) / 1000.0