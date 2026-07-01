from dataclasses import dataclass

@dataclass
class HexCell:
    """A specific H3 hexagonal area on the Earth's surface."""
    h3_id: str
    center_lat: float
    center_lon: float