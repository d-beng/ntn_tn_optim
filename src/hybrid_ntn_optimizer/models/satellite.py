from dataclasses import dataclass, field
from typing import List
from hybrid_ntn_optimizer.core.types import (
    ECIVector, 
    KeplerianElements, 
    OrbitType, 
    FrequencyBand
)
from hybrid_ntn_optimizer.models.beam import Beam

@dataclass(frozen=True)
class SatelliteState:
    """Instantaneous state of a satellite at a given epoch."""
    satellite_id: str
    epoch_utc: str           # ISO-8601
    position_eci: ECIVector  # metres
    velocity_eci: ECIVector  # m/s
    lat_deg: float
    lon_deg: float
    altitude_m: float
    active_beams: List[Beam] = field(default_factory=list)   

    @property
    def altitude_km(self) -> float:
        return self.altitude_m / 1_000.0

@dataclass
class SatelliteDescriptor:
    """
    Static identifier and orbital parameters for one satellite.
    """
    sat_id: str
    plane_index: int
    slot_index: int            
    elements: KeplerianElements
    orbit_type: OrbitType = OrbitType.LEO
    freq_band: FrequencyBand = FrequencyBand.KU
    eirp_dbw: float = 40.0    
    g_t_db: float = 10.0  
    max_spot_beams: int = 15 
     
    beam_radius_nadir_km: float = 200.0 
    max_steering_angle_deg: float = 45.0  

    def __repr__(self) -> str:
        return (
            f"SatelliteDescriptor(id={self.sat_id!r}, "
            f"plane={self.plane_index}, slot={self.slot_index}, "
            f"orbit={self.orbit_type.value})"
        )
    

