import h3
import math
from dataclasses import dataclass, field
from hybrid_ntn_optimizer.core.utils import haversine_distance
from typing import Any, List, Set
from enum import Enum

# We keep the Enum here or in a shared constants file to act as the routing key
class DeploymentScenario(Enum):
    UMA    = "UMA"
    UMI    = "UMI"
    RMA    = "RMA"
    INH    = "INH"
    INF_SH = "INF_SH"

@dataclass
class BaseStation:
    # 1. Identity & Geography
    bs_id: int
    lat: float
    lon: float
    scenario: DeploymentScenario  # Now this BS knows its 3GPP identity

    # 2. RF / Physical Layer Parameters (Fed from config)
    p_tx_dbm: float
    g_tx_dbi: float
    carrier_freq_hz: float
    total_bandwidth_hz: float
    capacity_mbps: float  
    bs_height_m: float
    min_user_dist_m: float
    interference_cutoff_m: float
    
    # 3. 3GPP Physics Parameters (Fed from config)
    shadow_sigma_los_db: float
    shadow_sigma_nlos_db: float
    
    # 4. Simulation State
    use_physical_radius: bool = False
    coverage_radius_km: float = 0.0
    center_h3_id: str = field(init=False)
    covered_h3_ids: Set[str] = field(default_factory=set)
    
    active_users: int = 0
    remaining_bandwidth_hz: float = field(init=False)         
    attached_users: List[Any] = field(default_factory=list)     
    remaining_capacity_mbps: float = field(init=False)
    
    def __post_init__(self):
        self.remaining_capacity_mbps = self.capacity_mbps
        self.remaining_bandwidth_hz = self.total_bandwidth_hz

    def set_resolution(self, resolution: int):
        self.center_h3_id = h3.latlng_to_cell(self.lat, self.lon, resolution)
        self.covered_h3_ids = {self.center_h3_id}
        
        if self.use_physical_radius and self.coverage_radius_km > 0:
            edge_len_km = h3.average_hexagon_edge_length(resolution, unit='km')
            k_rings = math.ceil(self.coverage_radius_km / edge_len_km)
            if k_rings > 0:
                candidate_hexes = h3.grid_disk(self.center_h3_id, k_rings)
                for hex_id in candidate_hexes:
                    h_lat, h_lon = h3.cell_to_latlng(hex_id)
                    dist = haversine_distance(self.lat, self.lon, h_lat, h_lon) / 1000.0
                    if dist <= self.coverage_radius_km:
                        self.covered_h3_ids.add(hex_id)