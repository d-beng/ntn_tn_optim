import h3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region, map_satellites_to_region




def build_h3_geojson(cells):
    """Converts H3 cells into a GeoJSON FeatureCollection for Plotly."""
    features = []
    for cell in cells:
        # H3 v4 returns (lat, lng)
        boundary_latlng = h3.cell_to_boundary(cell.h3_id)
        
        # GeoJSON strictly requires (lng, lat), so we swap them
        boundary_lnglat = [(lng, lat) for lat, lng in boundary_latlng]
        
        # GeoJSON polygons must be closed loops (first point == last point)
        boundary_closed = boundary_lnglat + [boundary_lnglat[0]]
        
        features.append({
            "type": "Feature",
            "id": cell.h3_id,
            "geometry": {"type": "Polygon", "coordinates": [boundary_closed]}
        })
    return {"type": "FeatureCollection", "features": features}


def plot_static_grid(region: Region, filename="fast_grid_test.html"):
    """Plots only the static H3 grid without any satellites or simulation."""
    print(f"Generating static plot for {len(region.cells)} cells...")
    
    # 1. Build the GeoJSON from the Region's cells
    # (Assuming build_h3_geojson is defined higher up in this file)
    geojson_hexes = build_h3_geojson(region.cells)
    
    # 2. Create a dummy dataframe just for Plotly to render the polygons
    df = pd.DataFrame([{"h3_id": cell.h3_id, "status": "Grid Cell"} for cell in region.cells])
    
    # 3. Render the Map
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_hexes,
        locations="h3_id",
        color="status",
        color_discrete_map={"Grid Cell": "rgba(0, 255, 255, 0.3)"}, # A nice translucent cyan
        mapbox_style="carto-darkmatter",
        center={"lat": 50.0, "lon": -85.0},  # Center on Ontario
        zoom=3.5,
        opacity=0.6,
        title=f"Static H3 Grid: {region.name} ({len(region.cells)} cells at Res {region.h3_resolution})"
    )
    
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, showlegend=False)
    fig.write_html(filename)
    print(f"Success! Grid map saved to {filename}")

def plot_hex_coverage_animation(leo: LEOConstellation, region: Region, duration_s: float, time_step_s: float, filename="ontario_coverage.html"):
    """Generates an animated map of the hexagonal beams over time and exports data to Excel."""
    print(f"Tessellating {region.name} into H3 Hexagons...")
    base_cells = region.cells
    geojson_hexes = build_h3_geojson(base_cells)
    
    print(f"Running physics engine for {len(base_cells)} cells over {duration_s}s...")
    
    all_data = []
    steps = int(duration_s / time_step_s)
    
    for step in range(steps + 1):
        dt_s = step * time_step_s
        print(f"Processing time step {dt_s:.1f}s / {duration_s:.1f}s", end="\r")
        
        # Ask the physics engine to map satellites to our ground cells
        if step == 0:
            active_beams = map_satellites_to_region(leo, region, dt_s, debug_log=True)
        else:
            active_beams = map_satellites_to_region(leo, region, dt_s, debug_log=False)
        covered_cell_ids = {beam.target_cell_id: beam for beam in active_beams}
        
        # Record the status of every cell at this specific second
        for cell in base_cells:
            beam = covered_cell_ids.get(cell.h3_id)
            is_covered = 1 if beam else 0
            sat_id = beam.satellite_id if beam else "NO SIGNAL"
            elev = f"{beam.elevation_deg:.1f}°" if beam else "N/A"
            
            all_data.append({
                "time_s": dt_s,
                "h3_id": cell.h3_id,
                "status": "Covered" if is_covered else "Gap",
                "satellite": sat_id,
                "elevation": elev,
                "color_val": is_covered
            })
            
    # Clear the print line
    print(" " * 50, end="\r")
    
    df = pd.DataFrame(all_data)
    
    # Calculate overall SLA (Service Level Agreement)
    worst_coverage = df.groupby("time_s")["color_val"].mean().min() * 100
    print(f"Minimum Constellation Coverage during simulation: {worst_coverage:.2f}%")

    # ==========================================
    # NEW: EXCEL EXPORT LOGIC
    # ==========================================
    excel_filename = filename.replace('.html', '_data.xlsx')
    print(f"Exporting dataset to {excel_filename}...")
    
    # Create a summary dataframe for the number of covered beams per timestamp
    summary_df = df.groupby('time_s').agg(
        total_beams=('h3_id', 'count'),
        covered_beams=('color_val', 'sum')
    ).reset_index()
    summary_df['coverage_percentage'] = (summary_df['covered_beams'] / summary_df['total_beams']) * 100

    # Write to a single Excel file with two sheets
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='Coverage_Summary', index=False)
        df.to_excel(writer, sheet_name='Detailed_Mapping', index=False)
    # ==========================================

    print("Rendering Plotly Animation (this may take a moment)...")
    
    # Build the animated Choropleth map
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_hexes,
        locations="h3_id",
        color="status",
        animation_frame="time_s",
        color_discrete_map={"Covered": "rgba(0, 255, 0, 0.5)", "Gap": "rgba(255, 0, 0, 0.5)"},
        category_orders={"status": ["Covered", "Gap"]}, 
        hover_name="satellite",
        hover_data={"h3_id": False, "status": False, "elevation": True, "time_s": False},
        mapbox_style="carto-darkmatter",
        center={"lat": 50.0, "lon": -85.0},  # Center on Ontario
        zoom=3.5,
        opacity=0.6,
        title=f"NTN Beam Coverage: {region.name} ({worst_coverage:.1f}% Min Coverage)"
    )
    
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig.write_html(filename)
    print(f"Success! Map saved to {filename}")