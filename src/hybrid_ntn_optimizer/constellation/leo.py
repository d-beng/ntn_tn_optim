from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from hybrid_ntn_optimizer.core.constants import (
    DEFAULT_EPOCH,
    DEFAULT_MIN_ELEVATION_DEG,
    DEFAULT_TIME_STEP_S,
)
from hybrid_ntn_optimizer.core.exceptions import ConstellationError
from hybrid_ntn_optimizer.core.types import (
    FrequencyBand,
    GeoPoint,
    OrbitType,
    WalkerParameters,
)
from hybrid_ntn_optimizer.constellation.walker_delta import (
    build_walker_delta,
)

from hybrid_ntn_optimizer.models.satellite import SatelliteDescriptor, SatelliteState

from hybrid_ntn_optimizer.constellation.propagator import (
    propagate_constellation,
    generate_ground_track,
)
from hybrid_ntn_optimizer.constellation.visibility import (
    CoverageCell,
    VisibilityRecord,
    best_satellite,
    coverage_fraction,
    coverage_snapshot,
    visible_satellites,
)


@dataclass
class LEOConstellation:


    params: WalkerParameters
    epoch_utc: str = DEFAULT_EPOCH
    name: str = "LEO-Shell"
    apply_j2: bool = True  # To Do: Make this more flexible in the future from dict input
    eirp_dbw: float = 40.0
    g_t_db: float = 10.0
    min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG
    max_spot_beams: int = 15 
    beam_radius_nadir_km: float = 120.0
    max_steering_angle_deg: float = 45.0
    descriptors: List[SatelliteDescriptor] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.descriptors:
            self.descriptors = build_walker_delta(
                params=self.params,
                freq_band=FrequencyBand.KU,
                eirp_dbw=self.eirp_dbw,
                g_t_db=self.g_t_db,
                name_prefix=self.name.upper().replace(" ", "-"),
                max_spot_beams=self.max_spot_beams,
                beam_radius_nadir_km=self.beam_radius_nadir_km,
                max_steering_angle_deg=self.max_steering_angle_deg
            )


    @classmethod
    def from_dict(cls, cfg: dict, epoch_utc: str = DEFAULT_EPOCH) -> "LEOConstellation":
        
        params = WalkerParameters(
            total_satellites=cfg["total_satellites"],
            num_planes=cfg["num_planes"],
            phasing=cfg["phasing"],
            inclination_deg=cfg["inclination_deg"],
            altitude_km=cfg["altitude_km"],
            orbit_type=OrbitType.LEO,
        )
        return cls(
            params=params,
            epoch_utc=epoch_utc,
            name=cfg.get("name", "Custom-LEO"),
            apply_j2=cfg.get("apply_j2", True),
            eirp_dbw=cfg.get("eirp_dbw", 40.0),
            g_t_db=cfg.get("g_t_db", 10.0),
            min_elevation_deg=cfg.get("min_elevation_deg", DEFAULT_MIN_ELEVATION_DEG),
            max_spot_beams=cfg.get("max_spot_beams", 15),
            beam_radius_nadir_km=cfg.get("beam_radius_nadir_km", 120.0),
            max_steering_angle_deg=cfg.get("max_steering_angle_deg", 45.0),
        )

    # ------------------------------------------------------------------
    # Core propagation interface
    # ------------------------------------------------------------------

    def snapshot(self, dt_s: float) -> List[SatelliteState]:
        return propagate_constellation(
            self.descriptors,
            self.epoch_utc,
            dt_s,
            apply_j2=self.apply_j2,
        )

    def ground_track(
        self,
        sat_id: str,
        duration_s: float,
        time_step_s: float = DEFAULT_TIME_STEP_S,
    ) -> List[SatelliteState]:
        
        desc = self._find_descriptor(sat_id)
        return generate_ground_track(
            desc,
            self.epoch_utc,
            duration_s,
            time_step_s=time_step_s,
            apply_j2=self.apply_j2,
        )

    # ------------------------------------------------------------------
    # Visibility / coverage
    # ------------------------------------------------------------------

    def visible_from(self, lat_deg: float, lon_deg: float, dt_s: float = 0.0):
        from hybrid_ntn_optimizer.constellation.propagator import build_earth_satellite
        
        states = self.snapshot(dt_s)
        
        # Build the EarthSatellite objects required by the new visibility logic
        earth_sats = [build_earth_satellite(d, self.epoch_utc) for d in self.descriptors]
        
        return visible_satellites(
            states,
            GeoPoint(lat_deg=lat_deg, lon_deg=lon_deg),
            self.min_elevation_deg,
            earth_sats=earth_sats
        )

    def best_satellite_from(
        self,
        lat_deg: float,
        lon_deg: float,
        dt_s: float = 0.0,
    ) -> Optional[VisibilityRecord]:
        """Highest-elevation visible satellite, or None."""
        # Reuse visible_from because it handles the Skyfield object creation 
        # and already sorts the results by elevation (highest first).
        vis = self.visible_from(lat_deg, lon_deg, dt_s)
        return vis[0] if vis else None

    def coverage_at(
        self,
        lat_grid: List[float],
        lon_grid: List[float],
        dt_s: float = 0.0,
    ) -> List[CoverageCell]:
       
        states = self.snapshot(dt_s)
        return coverage_snapshot(
            states, lat_grid, lon_grid, self.min_elevation_deg
        )

    def global_coverage_fraction(
        self,
        lat_step_deg: float = 5.0,
        lon_step_deg: float = 5.0,
        dt_s: float = 0.0,
    ) -> float:
        
        import numpy as _np

        lat_grid = list(_np.arange(-90.0, 90.0 + lat_step_deg, lat_step_deg))
        lon_grid = list(_np.arange(-180.0, 180.0 + lon_step_deg, lon_step_deg))
        cells = self.coverage_at(lat_grid, lon_grid, dt_s)
        return coverage_fraction(cells)

    # ------------------------------------------------------------------
    # Info / repr
    # ------------------------------------------------------------------

    @property
    def num_satellites(self) -> int:
        return len(self.descriptors)

    @property
    def altitude_km(self) -> float:
        return self.params.altitude_km

    @property
    def inclination_deg(self) -> float:
        return self.params.inclination_deg

    def __repr__(self) -> str:
        return (
            f"LEOConstellation(name={self.name!r}, "
            f"sats={self.num_satellites}, "
            f"alt={self.altitude_km:.0f} km, "
            f"inc={self.inclination_deg:.1f}°)"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_descriptor(self, sat_id: str) -> SatelliteDescriptor:
        for desc in self.descriptors:
            if desc.sat_id == sat_id:
                return desc
        raise ConstellationError(
            f"Satellite {sat_id!r} not found in constellation {self.name!r}."
        )