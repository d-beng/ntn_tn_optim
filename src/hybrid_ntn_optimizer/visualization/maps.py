import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
import plotly.graph_objects as go
import pandas as pd

def plot_global_constellation(leo: LEOConstellation, dt_s: float = 0.0, save_path: str = "constellation_map.png"):
    """Plots the instantaneous ground tracks of any provided LEO constellation."""
    print(f"Generating map for {leo.name}...")
    
    # Use the snapshot method from the leo object
    states = leo.snapshot(dt_s=dt_s)
    
    lats = [state.lat_deg for state in states]
    lons = [state.lon_deg for state in states]
    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.coastlines(linewidth=0.5)
    
    ax.scatter(lons, lats, color='red', s=5, transform=ccrs.PlateCarree(), 
               label=f"{leo.name} ({leo.num_satellites} sats)")
    
    plt.title(f"Global Coverage: {leo.name} (t = {dt_s}s)")
    plt.legend(loc='lower left')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Map successfully saved to {save_path}")
    plt.close() # Clean up memory


def plot_2d_interactive_animation(leo: LEOConstellation, duration_s: float, time_step_s: float, filename: str = "animated_2d_map.html"):
    """Generates an interactive animated 2D map showing satellite movement."""
    print(f"Creating 2D animation for {leo.name}...")
    
    frames = []
    steps = int(duration_s / time_step_s)
    
    # 1. Collect all simulation data
    all_data = []
    for step in range(steps + 1):
        dt_s = step * time_step_s
        states = leo.snapshot(dt_s=dt_s)
        for s in states:
            all_data.append({
                'id': s.satellite_id,
                'lat': s.lat_deg,
                'lon': s.lon_deg,
                'time': dt_s
            })
    
    df = pd.DataFrame(all_data)

    # 2. Build the Base Figure
    fig = go.Figure()

    # Add the initial frame (t=0)
    initial_df = df[df['time'] == 0]
    fig.add_trace(go.Scattergeo(
        lat=initial_df['lat'],
        lon=initial_df['lon'],
        mode='markers',
        marker=dict(size=4, color='red'),
        name="Satellites"
    ))

    # 3. Create Animation Frames
    frames = [
        go.Frame(
            data=[go.Scattergeo(lat=df[df['time'] == t]['lat'], lon=df[df['time'] == t]['lon'])],
            name=str(t)
        )
        for t in df['time'].unique()
    ]
    fig.frames = frames

    # 4. Add Play/Pause Buttons and Slider
    fig.update_layout(
        title=f"2D Satellite Movement: {leo.name}",
        geo=dict(showland=True, landcolor="rgb(243, 243, 243)"),
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 100, "redraw": True}}], "label": "Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}}], "label": "Pause", "method": "animate"}
            ],
            "type": "buttons"
        }]
    )

    fig.write_html(filename)
    print(f"2D Animation saved to {filename}")