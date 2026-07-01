from dataclasses import dataclass

@dataclass
class Beam:
    """An active RF connection between a satellite and a ground cell."""
    satellite_id: str
    target_cell_id: str
    elevation_deg: float
    slant_range_km: float
    is_active: bool = False
    
    # NEW: State tracking for NTN fair share
    active_users: int = 0
    allocated_mbps: float = 0.0