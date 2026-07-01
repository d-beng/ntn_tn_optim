from dataclasses import dataclass, field
from typing import Dict, Any, List
from hybrid_ntn_optimizer.models.cell import HexCell

@dataclass
class Region:
    """A geographic boundary for the simulation scenario."""
    name: str
    geojson_geometry: Dict[str, Any]
    h3_resolution: int
    cells: List[HexCell] = field(default_factory=list)