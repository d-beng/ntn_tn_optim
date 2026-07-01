import pandas as pd
import math
import h3
import plotly.express as px
import plotly.graph_objects as go
from omegaconf import OmegaConf, DictConfig

def build_h3_geojson(cells):
    """Generates GeoJSON polygons for H3 v4 cells."""
    features = []
    for cell in cells:
        # H3 v4 returns a tuple of (lat, lon).
        boundary = h3.cell_to_boundary(cell.h3_id)
        
        # GeoJSON requires (lon, lat), so we manually flip them!
        geojson_coords = [[lon, lat] for lat, lon in boundary]
        
        features.append({
            "type": "Feature",
            "id": cell.h3_id,
            "geometry": {"type": "Polygon", "coordinates": [geojson_coords]}
        })
    return {"type": "FeatureCollection", "features": features}

def build_bs_coverage_geojson(base_stations):
    """Generates accurate geographic polygon circles for 5G Base Station coverage."""
    features = []
    R_EARTH_KM = 6371.0
    
    for bs in base_stations:
        radius_km = bs.coverage_radius_km
        coords = []
        for i in range(65):
            angle = math.pi * 2 * i / 64
            dx = radius_km * math.cos(angle)
            dy = radius_km * math.sin(angle)
            
            new_lat = bs.lat + (dy / R_EARTH_KM) * (180 / math.pi)
            new_lon = bs.lon + (dx / R_EARTH_KM) * (180 / math.pi) / math.cos(bs.lat * math.pi / 180)
            coords.append([new_lon, new_lat])
            
        features.append({
            "type": "Feature",
            "id": f"BS_{bs.bs_id}",
            "geometry": {"type": "Polygon", "coordinates": [coords]}
        })
    return {"type": "FeatureCollection", "features": features}
"""
def plot_master_hybrid_animation(region, users, base_stations, beam_data, duration_s, time_step_s, filename="hybrid_master_map.html"):
    print(f"\n🎨 Rendering Master Interactive Visualization...")
    
    hex_geojson = build_h3_geojson(region.cells)
    bs_coverage_geojson = build_bs_coverage_geojson(base_stations)
    
    # Build a complete DataFrame with every hex at every timestep (required for smooth Plotly animation)
    all_frames = []
    time_steps = list(range(0, duration_s + time_step_s, time_step_s))
    
    for t_s in time_steps:
        # Find beams active right now
        active_now = {b["h3_id"]: b for b in beam_data if b["time_s"] == t_s}
        
        for cell in region.cells:
            if cell.h3_id in active_now:
                b = active_now[cell.h3_id]
                all_frames.append({
                    "Time_Hour": f"Hour {t_s / 3600.0:.1f}",
                    "h3_id": cell.h3_id,
                    "status": "NTN Active Beam",
                    "satellite": b["satellite"],
                    "elevation": f"{b['elevation']}°"
                })
            else:
                all_frames.append({
                    "Time_Hour": f"Hour {t_s / 3600.0:.1f}",
                    "h3_id": cell.h3_id,
                    "status": "NTN Standby",
                    "satellite": "None",
                    "elevation": "N/A"
                })
                
    df = pd.DataFrame(all_frames)
    
    # 1. Base Map (Animated Hexagons)
    fig = px.choropleth_mapbox(
        df, geojson=hex_geojson, locations="h3_id", color="status",
        animation_frame="Time_Hour",
        color_discrete_map={
            "NTN Active Beam": "rgba(0, 255, 100, 0.5)",  # Bright Green
            "NTN Standby": "rgba(50, 50, 50, 0.1)"        # Barely visible grey
        },
        hover_name="satellite", hover_data={"h3_id": False, "status": False, "Time_Hour": False},
        mapbox_style="carto-darkmatter", zoom=4.5,
        center={"lat": region.cells[0].center_lat, "lon": region.cells[0].center_lon}, 
        title="Hybrid NTN-TN Real-Time Traffic Routing"
    )
    
    # 2. Add Users (Upgraded to be highly visible and color-coded)
    tn_users = [u for u in users if u.coverage_type == "TN"]
    
    leo_users = [u for u in users if u.coverage_type == "LEO"]
    print(f"Plotting {len(tn_users)} TN users and {len(leo_users)} LEO users on the map...")
    # Plot TN Users (Cyan)
    if tn_users:
        fig.add_trace(go.Scattermapbox(
            lat=[u.home_lat for u in tn_users], 
            lon=[u.home_lon for u in tn_users],
            mode='markers', 
            marker=dict(size=5, color='cyan', opacity=0.8),
            name='TN Users', 
            hoverinfo='text',
            text=[f"User {u.user_id}<br>Type: TN<br>Demand: {u.base_demand_mbps:.1f} Mbps" for u in tn_users]
        ))

    # Plot NTN Users (Magenta)
    if leo_users:
        fig.add_trace(go.Scattermapbox(
            lat=[u.home_lat for u in leo_users], 
            lon=[u.home_lon for u in leo_users],
            mode='markers', 
            marker=dict(size=5, color='magenta', opacity=0.8),
            name='NTN Users', 
            hoverinfo='text',
            text=[f"User {u.user_id}<br>Type: NTN<br>Demand: {u.base_demand_mbps:.1f} Mbps" for u in leo_users]
        ))
    # 3. Add 5G Towers (Orange dots)
    fig.add_trace(go.Scattermapbox(
        lat=[bs.lat for bs in base_stations], lon=[bs.lon for bs in base_stations],
        mode='markers', marker=dict(size=10, color='orange', symbol='circle'),
        name='5G Base Stations', hoverinfo='text',
        text=[f"Tower {bs.bs_id}<br>Radius: {bs.coverage_radius_km:.2f} km" for bs in base_stations]
    ))
    
    # 4. Inject physical 5G coverage footprints (Orange Translucent Circles)
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        mapbox=dict(
            layers=[dict(
                source=bs_coverage_geojson,
                type="fill",
                color="rgba(255, 165, 0, 0.25)" 
            )]
        )
    )
    
    fig.write_html(filename)
    print(f"✅ Master Visualization saved to {filename}")
"""

def plot_master_hybrid_animation_OLD(region, users, base_stations, beam_data, user_data, duration_s, time_step_s, filename="Final_Animation.html"):
    print(f"\n🎨 Rendering Unified Visualization (This may take a moment to compile...)")
    
    hex_geojson = build_h3_geojson(region.cells)
    bs_coverage_geojson = build_bs_coverage_geojson(base_stations)
    
    # 1. Setup Data & Color Mappings
    time_steps = list(range(0, duration_s + time_step_s, time_step_s))
    
    user_color_map = {"TN": "deepskyblue", "LEO": "hotpink", "DROPPED": "red", "IDLE": "gray"}
    hex_z_map = {"Standby": 0, "Active": 1}
    hex_colorscale = [[0, "rgba(50, 50, 50, 0.1)"], [1, "rgba(0, 255, 100, 0.5)"]]
    
    # Pre-extract all H3 IDs so the map doesn't jitter
    all_h3_ids = [cell.h3_id for cell in region.cells]
    
    fig = go.Figure()

    # ==========================================
    # 2. DRAW BASE TRACES (Hour 0)
    # ==========================================
    # Trace 0: The Hexagons (Choropleth)
    initial_beams = [b["h3_id"] for b in beam_data if b["time_s"] == 0]
    initial_z = [1 if h3_id in initial_beams else 0 for h3_id in all_h3_ids]
    
    fig.add_trace(go.Choroplethmapbox(
        geojson=hex_geojson, locations=all_h3_ids, z=initial_z,
        colorscale=hex_colorscale, zmin=0, zmax=1,
        marker_opacity=0.6, marker_line_width=1, showscale=False,
        name="Satellite Beams", hoverinfo="skip"
    ))

    # Trace 1: The Users (Scatter)
    initial_users = [u for u in user_data if u["Hour"] == "Hour 0.0"]
    initial_u_colors = [user_color_map[u["State"]] for u in initial_users]
    
    fig.add_trace(go.Scattermapbox(
        lat=[u["Lat"] for u in initial_users], lon=[u["Lon"] for u in initial_users],
        mode='markers', marker=dict(size=6, color=initial_u_colors, opacity=0.9),
        name='Mobile Users', hoverinfo='text',
        text=[f"User {u['User_ID']}<br>State: {u['State']}" for u in initial_users]
    ))

    # Trace 2: The 5G Towers (Static Scatter)
    fig.add_trace(go.Scattermapbox(
        lat=[bs.lat for bs in base_stations], lon=[bs.lon for bs in base_stations],
        mode='markers', marker=dict(size=10, color='orange', symbol='circle'),
        name='5G Base Stations', hoverinfo='text',
        text=[f"Tower {bs.bs_id}<br>Radius: {bs.coverage_radius_km:.2f} km" for bs in base_stations]
    ))

    # ==========================================
    # 3. BUILD THE ANIMATION FRAMES
    # ==========================================
    frames = []
    slider_steps = []
    
    for t_s in time_steps:
        hour_str = f"Hour {t_s / 3600.0:.1f}"
        
        # Calculate Frame Data
        active_beams = [b["h3_id"] for b in beam_data if b["time_s"] == t_s]
        frame_z = [1 if h3_id in active_beams else 0 for h3_id in all_h3_ids]
        
        frame_users = [u for u in user_data if u["Hour"] == hour_str]
        frame_u_colors = [user_color_map[u["State"]] for u in frame_users]
        frame_u_text = [f"User {u['User_ID']}<br>State: {u['State']}" for u in frame_users]
        
        # Inject the updated data into the frame (Traces 0 and 1)
        frame = go.Frame(
            name=hour_str,
            data=[
                go.Choroplethmapbox(z=frame_z),
                go.Scattermapbox(marker=dict(color=frame_u_colors), text=frame_u_text)
            ],
            traces=[0, 1] # Explicitly tell Plotly to only update the Hexagons and Users!
        )
        frames.append(frame)
        
        # Build the slider tick
        slider_steps.append({
            "args": [[hour_str], {"frame": {"duration": 800, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
            "label": hour_str, "method": "animate"
        })

    fig.frames = frames

    # ==========================================
    # 4. FINAL LAYOUT & UI
    # ==========================================
    fig.update_layout(
        title="Hybrid NTN-TN Real-Time Traffic Routing (Unified Visualization)",
        mapbox=dict(
            style="carto-darkmatter",
            center={"lat": 46.0, "lon": -80.0}, zoom=4.5,
            layers=[dict(source=bs_coverage_geojson, type="fill", color="rgba(255, 165, 0, 0.25)")]
        ),
        margin={"r":0,"t":50,"l":0,"b":0},
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}], "label": "Play ▶", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}], "label": "Pause ⏸", "method": "animate"}
            ],
            "direction": "left", "pad": {"r": 10, "t": 87}, "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
        }],
        sliders=[{"active": 0, "yanchor": "top", "xanchor": "left", "currentvalue": {"font": {"size": 20}, "prefix": "Time: ", "visible": True, "xanchor": "right"}, "transition": {"duration": 0}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0, "steps": slider_steps}]
    )
    
    fig.write_html(filename)
    print(f"✅ Master Visualization saved to {filename}")



def plot_master_hybrid_animation(region, users, base_stations, beam_data, user_data, duration_s, time_step_s, filename="Final_Animation.html"):
    print(f"\n🎨 Rendering Unified God-Mode Visualization (Compiling frames...)")
    
    hex_geojson = build_h3_geojson(region.cells)
    bs_coverage_geojson = build_bs_coverage_geojson(base_stations)
    
    time_steps = list(range(0, duration_s + time_step_s, time_step_s))
    all_h3_ids = [cell.h3_id for cell in region.cells]
    
    fig = go.Figure()

    # ==========================================
    # 1. DRAW BASE TRACES (Hour 0)
    # ==========================================
    
    # Trace 0: The Hexagons (Choropleth for Satellite Beams)
    initial_beams = [b["h3_id"] for b in beam_data if b["time_s"] == 0]
    initial_z = [1 if h3_id in initial_beams else 0 for h3_id in all_h3_ids]
    
    fig.add_trace(go.Choroplethmapbox(
        geojson=hex_geojson, locations=all_h3_ids, z=initial_z,
        colorscale=[[0, "rgba(50, 50, 50, 0.1)"], [1, "rgba(0, 255, 100, 0.5)"]], 
        zmin=0, zmax=1, marker_opacity=0.6, marker_line_width=1, showscale=False,
        name="Satellite Beams", hoverinfo="skip"
    ))

    # Traces 1-4: The Users (Separated to eliminate the string interpolation bug)
    user_states = [
        ("TN", "deepskyblue", "TN Served (5G)"),
        ("LEO", "hotpink", "NTN Served (Satellite)"),
        ("DROPPED", "red", "Dropped (Outage)"),
        ("IDLE", "gray", "Idle")
    ]
    
    initial_users = [u for u in user_data if u["Hour"] == "Hour 0.0"]
    for state_id, color, label in user_states:
        state_users = [u for u in initial_users if u["State"] == state_id]
        fig.add_trace(go.Scattermapbox(
            lat=[u["Lat"] for u in state_users], 
            lon=[u["Lon"] for u in state_users],
            mode='markers', marker=dict(size=6, color=color, opacity=0.9),
            name=label, hoverinfo='text',
            text=[f"User {u['User_ID']}<br>State: {state_id}" for u in state_users]
        ))

    # Trace 5: The 5G Towers (Static Reference)
    fig.add_trace(go.Scattermapbox(
        lat=[bs.lat for bs in base_stations], lon=[bs.lon for bs in base_stations],
        mode='markers', marker=dict(size=10, color='orange', symbol='circle'),
        name='5G Base Stations', hoverinfo='text',
        text=[f"Tower {bs.bs_id}<br>Radius: {bs.coverage_radius_km:.2f} km" for bs in base_stations]
    ))

    # ==========================================
    # 2. BUILD THE ANIMATION FRAMES
    # ==========================================
    frames = []
    slider_steps = []
    
    for t_s in time_steps:
        hour_str = f"Hour {t_s / 3600.0:.1f}"
        
        # Synchronize active beams for the current frame
        active_beams = [b["h3_id"] for b in beam_data if b["time_s"] == t_s]
        frame_z = [1 if h3_id in active_beams else 0 for h3_id in all_h3_ids]
        
        frame_data = [go.Choroplethmapbox(z=frame_z)]
        
        # Distribute users into their respective states for the current frame
        frame_users = [u for u in user_data if u["Hour"] == hour_str]
        for state_id, _, _ in user_states:
            state_users = [u for u in frame_users if u["State"] == state_id]
            frame_data.append(go.Scattermapbox(
                lat=[u["Lat"] for u in state_users],
                lon=[u["Lon"] for u in state_users],
                text=[f"User {u['User_ID']}<br>State: {state_id}" for u in state_users]
            ))
            
        frame = go.Frame(
            name=hour_str,
            data=frame_data,
            traces=[0, 1, 2, 3, 4]  # Explicit target tracking for variable arrays
        )
        frames.append(frame)
        
        slider_steps.append({
            "args": [[hour_str], {"frame": {"duration": 800, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
            "label": hour_str, "method": "animate"
        })

    fig.frames = frames

    # ==========================================
    # 3. CONFIGURE STRUCTURAL LAYERS & UI
    # ==========================================
    mapbox_layers = [
        # Layer 1: Translucent 5G Footprints
        dict(
            source=bs_coverage_geojson, 
            type="fill", 
            color="rgba(255, 165, 0, 0.25)"
        )
    ]
    
    # Layer 2: In-memory Ontario Administrative Borders
    if hasattr(region, 'geojson_geometry') and region.geojson_geometry:        
        # FIX: Check if it's a Hydra DictConfig container, and convert it to a raw Python dict!
        raw_geometry = region.geojson_geometry
        if isinstance(raw_geometry, DictConfig):
            raw_geometry = OmegaConf.to_container(raw_geometry, resolve=True)
        mapbox_layers.append(dict(
            source=raw_geometry,
            type="line",
            color="cyan",
            line=dict(width=2)
        ))
    else:
        print("⚠️ Warning: region.geojson_geometry missing or unreadable. Boundary lines omitted.")

    fig.update_layout(
        title="Hybrid NTN-TN Real-Time Traffic Routing (System Level Engine)",
        mapbox=dict(
            style="carto-darkmatter",
            center={"lat": 46.0, "lon": -80.0}, zoom=4.5,
            layers=mapbox_layers
        ),
        margin={"r":0,"t":50,"l":0,"b":0},
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}], "label": "Play ▶", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}], "label": "Pause ⏸", "method": "animate"}
            ],
            "direction": "left", "pad": {"r": 10, "t": 87}, "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
        }],
        sliders=[{"active": 0, "yanchor": "top", "xanchor": "left", "currentvalue": {"font": {"size": 20}, "prefix": "Time: ", "visible": True, "xanchor": "right"}, "transition": {"duration": 0}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0, "steps": slider_steps}]
    )
    
    fig.write_html(filename)
    print(f"✅ Master Visualization compiled successfully and saved to {filename}")