import h3
import logging
from typing import List
from omegaconf import DictConfig, OmegaConf 

from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.models.cell import HexCell
from hybrid_ntn_optimizer.models.beam import Beam

log = logging.getLogger(__name__)
"""
#Takes a complex GeoJSON Region (like a real country border) and perfectly fills it with H3 HexCells.

def tessellate_region(region: Region) -> List[HexCell]:

    geom = region.geojson_geometry
    
    if isinstance(geom, DictConfig):
        geom = OmegaConf.to_container(geom, resolve=True)
        
    geom_type = geom.get("type", "")
    polygons = []
    
    if geom_type == "Polygon":
        polygons = [geom]
    elif geom_type == "MultiPolygon":
        polygons = [{"type": "Polygon", "coordinates": coords} for coords in geom["coordinates"]]
    
    cell_ids = set()
    for poly in polygons:
        cells = h3.geo_to_cells(poly, res=region.h3_resolution)
        cell_ids.update(cells)
            
    region.cells = []
    for cid in cell_ids:
        lat, lon = h3.cell_to_latlng(cid)
        region.cells.append(HexCell(h3_id=cid, center_lat=lat, center_lon=lon))
        
    return region.cells
"""
def tessellate_region(region: Region, pad_edges: bool = True) -> None:
    """
    Takes a GeoJSON Region and fills it with H3 HexCells.
    If pad_edges is True, it adds a 1-cell buffer around the perimeter 
    to ensure 100% coverage despite H3's strict Centroid Rule.
    """
    geom = region.geojson_geometry
    
    if isinstance(geom, DictConfig):
        geom = OmegaConf.to_container(geom, resolve=True)
        
    geom_type = geom.get("type", "")
    polygons = []
    
    if geom_type == "Polygon":
        polygons = [geom]
    elif geom_type == "MultiPolygon":
        polygons = [{"type": "Polygon", "coordinates": coords} for coords in geom["coordinates"]]
    
    # 1. Get the strict internal cells (The 69 cells)
    base_cell_ids = set()
    for poly in polygons:
        # Note: If you are using H3 v4, this might be h3.polygon_to_cells
        cells = h3.geo_to_cells(poly, res=region.h3_resolution) 
        base_cell_ids.update(cells)
            
    # 2. THE FIX: Pad the edges
    final_cell_ids = set(base_cell_ids)
    if pad_edges:
        for cid in base_cell_ids:
            # h3.k_ring (v3) or h3.grid_disk (v4) gets the cell + its immediate neighbors
            neighbors = h3.grid_disk(cid, 1) 
            final_cell_ids.update(neighbors)
            
    # 3. Attach to the region
    region.cells = []
    for cid in final_cell_ids:
        lat, lon = h3.cell_to_latlng(cid)
        region.cells.append(HexCell(h3_id=cid, center_lat=lat, center_lon=lon))
        
    print(f"Generated {len(region.cells)} cells (including edge padding).")


"""
def map_satellites_to_region(
    leo: LEOConstellation, 
    region: Region,  
    dt_s: float = 0.0
) -> List[Beam]:
    active_beams = []
    
    for cell in region.cells:
        # Ask your existing Space engine for the best link
        best_sat = leo.best_satellite_from(lat_deg=cell.center_lat, lon_deg=cell.center_lon, dt_s=dt_s)
        
        if best_sat:
            active_beams.append(Beam(
                satellite_id=best_sat.satellite_id,
                target_cell_id=cell.h3_id,
                elevation_deg=best_sat.elevation_deg,
                slant_range_km=best_sat.slant_range_km,
                is_active=True
            ))
            
    return active_beams
"""

"""
#ptimized Global Assignment: Gathers all possible links in the region, sorts them by best elevation, and assigns beams top-down while respecting the satellite hardware limits (max_spot_beams).
def map_satellites_to_region(
    leo: LEOConstellation, 
    region: Region,  
    dt_s: float = 0.0
) -> List[Beam]:
   
    active_beams = []
    satellite_beam_usage = {}
    covered_cells = set()
    
    # 1. GATHER ALL POTENTIAL LINKS
    all_possible_links = []
    
    for cell in region.cells:
        visible_sats = leo.visible_from(lat_deg=cell.center_lat, lon_deg=cell.center_lon, dt_s=dt_s)
        
        for candidate_sat in visible_sats:
            all_possible_links.append({
                "cell_id": cell.h3_id,
                "sat_id": candidate_sat.satellite_id,
                "elevation_deg": candidate_sat.elevation_deg,
                "slant_range_km": candidate_sat.slant_range_km
            })
            
    # 2. SORT GLOBALLY (Highest Elevation First)
    # We prioritize the strongest, most direct RF links across the whole map.
    all_possible_links.sort(key=lambda x: x["elevation_deg"], reverse=True)
    
    # 3. ASSIGN BEAMS FROM THE TOP DOWN
    for link in all_possible_links:
        cell_id = link["cell_id"]
        sat_id = link["sat_id"]
        
        # If this cell already got assigned a better beam earlier in the loop, skip it
        if cell_id in covered_cells:
            continue
            
        # Look up hardware limits using the helper method we added to leo.py
        try:
            sat_hardware = leo.get_descriptor(sat_id)
            max_beams = sat_hardware.max_spot_beams
        except AttributeError:
            max_beams = 4  # Safe fallback just in case
            
        current_usage = satellite_beam_usage.get(sat_id, 0)
        
        # If the satellite still has a free beam, make the official assignment!
        if current_usage < max_beams:
            active_beams.append(Beam(
                satellite_id=sat_id,
                target_cell_id=cell_id,
                elevation_deg=link["elevation_deg"],
                slant_range_km=link["slant_range_km"],
                is_active=True
            ))
            
            # Mark the cell as covered and increment the satellite's beam counter
            covered_cells.add(cell_id)
            satellite_beam_usage[sat_id] = current_usage + 1
            
            # Optimization: If we have covered every single cell, we can exit early!
            if len(covered_cells) == len(region.cells):
                break
                
    return active_beams
"""

"""
    Maximizes Total Coverage (Least-Flexible First Algorithm):
    Prioritizes ground cells that have the fewest satellite options to prevent 
    highly-connected cells from stealing bottlenecked resources.
"""
"""
def map_satellites_to_region(
    leo: LEOConstellation, 
    region: Region,  
    dt_s: float = 0.0
) -> List[Beam]:
   
    active_beams = []
    satellite_beam_usage = {}
    
    # 1. GATHER ALL OPTIONS PER CELL
    # List format: [{"cell_id": str, "options": [Sat1, Sat2, ...]}, ...]
    cell_visibility_map = []
    
    for cell in region.cells:
        visible_sats = leo.visible_from(lat_deg=cell.center_lat, lon_deg=cell.center_lon, dt_s=dt_s)
        
        if visible_sats:
            # We sort the local options for this specific cell by elevation
            # so if it has to pick, it picks its own best signal
            sorted_local_options = sorted(visible_sats, key=lambda s: s.elevation_deg, reverse=True)
            
            cell_visibility_map.append({
                "cell_id": cell.h3_id,
                "options": sorted_local_options
            })
            
    # 2. SORT GLOBALLY BY FEWEST OPTIONS (The Max Coverage Secret!)
    # Cells that can only see 1 satellite will be at the front of the list.
    # Cells that can see 10 satellites will be at the back.
    cell_visibility_map.sort(key=lambda x: len(x["options"]))
    
    # 3. ASSIGN BEAMS
    for target in cell_visibility_map:
        cell_id = target["cell_id"]
        assigned = False
        
        for candidate_sat in target["options"]:
            sat_id = candidate_sat.satellite_id
            
            try:
                sat_hardware = leo.get_descriptor(sat_id)
                max_beams = sat_hardware.max_spot_beams
            except AttributeError:
                max_beams = 4 
                
            current_usage = satellite_beam_usage.get(sat_id, 0)
            
            # If this satellite has a free beam, take it!
            if current_usage < max_beams:
                active_beams.append(Beam(
                    satellite_id=sat_id,
                    target_cell_id=cell_id,
                    elevation_deg=candidate_sat.elevation_deg,
                    slant_range_km=candidate_sat.slant_range_km,
                    is_active=True
                ))
                
                satellite_beam_usage[sat_id] = current_usage + 1
                assigned = True
                
                # We found a beam for this cell, stop looking and move to the next cell!
                break 
                
    return active_beams
"""



def map_satellites_to_region(
    leo: LEOConstellation, 
    region: Region,  
    dt_s: float = 0.0,
    debug_log: bool = False  # <--- NEW: Debug toggle
) -> List[Beam]:
    """
    Maximizes Total Coverage (Least-Flexible First Algorithm).
    Includes a debug logging feature to verify beam assignments.
    """
    active_beams = []
    satellite_beam_usage = {}
    cell_visibility_map = []
    
    # 1. GATHER ALL OPTIONS PER CELL
    for cell in region.cells:
        visible_sats = leo.visible_from(lat_deg=cell.center_lat, lon_deg=cell.center_lon, dt_s=dt_s)
        if visible_sats:
            sorted_local_options = sorted(visible_sats, key=lambda s: s.elevation_deg, reverse=True)
            cell_visibility_map.append({
                "cell_id": cell.h3_id,
                "options": sorted_local_options
            })
            
    # 2. SORT GLOBALLY BY FEWEST OPTIONS 
    cell_visibility_map.sort(key=lambda x: len(x["options"]))
    
    if debug_log:
        print(f"\n{'='*50}")
        print(f"TIMESTEP: {dt_s}s | REGION: {region.name}")
        print(f"{'='*50}")

    # 3. ASSIGN BEAMS
    for target in cell_visibility_map:
        cell_id = target["cell_id"]
        options = target["options"]
        assigned = False
        
        if debug_log:
            print(f"\nEvaluating Cell: {cell_id} | Visible Satellites: {len(options)}")
        
        for candidate_sat in options:
            sat_id = candidate_sat.satellite_id
            
            try:
                sat_hardware = leo._find_descriptor(sat_id)
                max_beams = sat_hardware.max_spot_beams
            except AttributeError:
                max_beams = 4 
                print(sat_hardware.max_spot_beams)
                
            current_usage = satellite_beam_usage.get(sat_id, 0)
            
            if debug_log:
                print(f"  -> Checking {sat_id}: {current_usage}/{max_beams} beams used (Elev: {candidate_sat.elevation_deg:.1f}°)")
            
            # If this satellite has a free beam, take it!
            if current_usage < max_beams:
                active_beams.append(Beam(
                    satellite_id=sat_id,
                    target_cell_id=cell_id,
                    elevation_deg=candidate_sat.elevation_deg,
                    slant_range_km=candidate_sat.slant_range_km,
                    is_active=True
                ))
                
                satellite_beam_usage[sat_id] = current_usage + 1
                assigned = True
                
                if debug_log:
                    print(f"  => [ASSIGNED] {sat_id} locked to {cell_id}")
                
                break 
                
        if debug_log and not assigned:
            print(f"  => [DROPPED] ALL visible satellites for {cell_id} are full!")
                
    if debug_log:
        print(f"\n{'='*50}")
        print(f"SUMMARY: {len(active_beams)} Beams Assigned | {len(region.cells) - len(active_beams)} Cells Dropped")
        print(f"{'='*50}\n")
        
    return active_beams