"""
hybrid_ntn_optimizer.constellation
===================================

Satellite constellation modeling: geometry, propagation, and visibility.

Typical usage (Starlink default)
---------------------------------
>>> from hybrid_ntn_optimizer.constellation import LEOConstellation
>>> leo = LEOConstellation.starlink_shell1()
>>> states = leo.snapshot(dt_s=0)           # all 1 584 satellites at t=0
>>> vis = leo.visible_from(45.4, -75.7)     # Ottawa, Ontario
>>> print(vis[0].elevation_deg)

Custom constellation
--------------------
>>> from hybrid_ntn_optimizer.core.types import WalkerParameters, OrbitType
>>> params = WalkerParameters(
...     total_satellites=648, num_planes=18, phasing=1,
...     inclination_deg=87.9, altitude_km=1200.0, orbit_type=OrbitType.LEO,
... )
>>> oneweb = LEOConstellation(params=params, name="OneWeb-Shell-1")
"""

from hybrid_ntn_optimizer.constellation.base import ConstellationBase
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.constellation.propagator import (
    propagate_constellation,
    propagate_satellite,
    generate_ground_track,
    iso8601_to_jd,
    advance_epoch,
)
from hybrid_ntn_optimizer.constellation.visibility import (
    check_visibility,
    visible_satellites,
    best_satellite,
    coverage_snapshot,
    coverage_fraction,
    CoverageCell,
    instantaneous_coverage_radius_km,
)
from hybrid_ntn_optimizer.constellation.walker_delta import (
    build_walker_delta,
)

__all__ = [
    # Base / abstract
    "ConstellationBase",
    # Concrete constellation class
    "LEOConstellation",
    # Walker geometry
    "build_walker_delta",
    # Propagation
    "propagate_satellite",
    "propagate_constellation",
    "generate_ground_track",
    "iso8601_to_jd",
    "advance_epoch",
    # Visibility
    "check_visibility",
    "visible_satellites",
    "best_satellite",
    "coverage_snapshot",
    "coverage_fraction",
    "CoverageCell",
    "instantaneous_coverage_radius_km",
]