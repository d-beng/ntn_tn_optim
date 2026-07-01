import numpy as np
import plotly.graph_objects as go
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation

def plot_3d_interactive_globe(leo: LEOConstellation, dt_s: float = 0.0, filename: str = "interactive_constellation.html"):
    """Generates an interactive 3D HTML globe for the provided constellation."""
    print(f"Generating 3D interactive globe for {leo.name}...")
    
    states = leo.snapshot(dt_s=dt_s) #
    
    x_vals = [state.position_eci.x / 1000.0 for state in states]
    y_vals = [state.position_eci.y / 1000.0 for state in states]
    z_vals = [state.position_eci.z / 1000.0 for state in states]
    
    # Simple Earth Sphere
    R_earth = 6371.0 
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    x_earth = R_earth * np.cos(u) * np.sin(v)
    y_earth = R_earth * np.sin(u) * np.sin(v)
    z_earth = R_earth * np.cos(v)
    
    fig = go.Figure()
    fig.add_surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.8)
    
    fig.add_scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='markers',
                      marker=dict(size=3, color='red'), name=leo.name)
    
    fig.update_layout(title=f"3D Visualization: {leo.name}", template='plotly_dark')
    
    fig.write_html(filename)
    print(f"Interactive 3D map saved to {filename}")

def plot_3d_animated_globe(leo: LEOConstellation, duration_s: float, time_step_s: float, filename: str = "animated_3d_globe.html"):
    """Generates an interactive animated 3D globe showing orbital movement."""
    print(f"Creating 3D animation for {leo.name}...")
    
    steps = int(duration_s / time_step_s)
    
    # 1. Earth Surface Math
    R_earth = 6371.0
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x_earth = R_earth * np.cos(u) * np.sin(v)
    y_earth = R_earth * np.sin(u) * np.sin(v)
    z_earth = R_earth * np.cos(v)

    # 2. Create Figure and Frames
    fig = go.Figure(
        data=[
            go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', opacity=0.8, showscale=False),
            go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(size=3, color='red'))
        ],
        layout=go.Layout(
            updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]
        )
    )

    frames = []
    for step in range(steps + 1):
        dt_s = step * time_step_s
        states = leo.snapshot(dt_s=dt_s)
        x = [s.position_eci.x / 1000.0 for s in states]
        y = [s.position_eci.y / 1000.0 for s in states]
        z = [s.position_eci.z / 1000.0 for s in states]
        
        frames.append(go.Frame(
            data=[go.Surface(), go.Scatter3d(x=x, y=y, z=z)],
            name=str(dt_s)
        ))

    fig.frames = frames
    fig.update_layout(scene=dict(aspectmode='data'), template="plotly_dark")
    fig.write_html(filename)
    print(f"3D Animation saved to {filename}")