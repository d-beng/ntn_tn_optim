import json
import folium
from folium.plugins import FastMarkerCluster
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# Core Models
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region
from hybrid_ntn_optimizer.traffic.profiles import generate_users
from hybrid_ntn_optimizer.terrestrial.coverage import generate_terrestrial_network

def generate_html_map(users, base_stations, output_file="Deployment_Preview.html"):
    print(f"\n[Map Generation] Rendering {len(base_stations)} Towers and {len(users)} Users...", flush=True)
    
    # Calculate center of the map based on users
    avg_lat = sum(u.home_lat for u in users) / len(users)
    avg_lon = sum(u.home_lon for u in users) / len(users)
    
    # Initialize Folium Map (CartoDB positron is great for seeing data overlays)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6, tiles="CartoDB positron")

    # 1. Plot Users (Using FastMarkerCluster to prevent browser freezing)
    print("  -> Plotting users...", flush=True)
    
    # Subsample users to prevent massive 500MB+ HTML files
    MAX_USERS_TO_PLOT = 25000
    if len(users) > MAX_USERS_TO_PLOT:
        step = len(users) // MAX_USERS_TO_PLOT
        users_to_plot = users[::step]
        print(f"  -> Subsampling users for preview map (showing {len(users_to_plot):,} out of {len(users):,} to keep file size small)...", flush=True)
    else:
        users_to_plot = users

    user_data = [[u.home_lat, u.home_lon] for u in users_to_plot]
    FastMarkerCluster(user_data, name="Simulated Users").add_to(m)

    # 2. Plot Base Stations by Tier
    print("  -> Plotting base stations...", flush=True)
    umi_group = folium.FeatureGroup(name="UMI (Small Cells) - Red", show=True)
    uma_group = folium.FeatureGroup(name="UMA (Macro Cells) - Blue", show=True)
    rma_group = folium.FeatureGroup(name="RMA (Rural Cells) - Green", show=True)

    for bs in base_stations:
        tier = bs.scenario.name
        
        # Color coding and opacity based on tier
        if tier == "UMI":
            color = "#d62728" # Red
            group = umi_group
            fill_opacity = 0.4
        elif tier == "UMA":
            color = "#1f77b4" # Blue
            group = uma_group
            fill_opacity = 0.2
        else: # RMA
            color = "#2ca02c" # Green
            group = rma_group
            fill_opacity = 0.1

        # Tooltip text showing Base Station stats
        popup_text = (
            f"<b>Tower ID:</b> {bs.bs_id}<br>"
            f"<b>Tier:</b> {tier}<br>"
            f"<b>Radius:</b> {bs.coverage_radius_km:.2f} km<br>"
            f"<b>Assigned Users:</b> {bs.assigned_user_count}<br>"
            f"<b>Density:</b> {bs.cluster_density:.1f} users/km²"
        )

        # Plot the tower center dot
        folium.CircleMarker(
            location=[bs.lat, bs.lon],
            radius=2,
            color="black",
            weight=1,
            fill=True,
            fill_color="black",
            fill_opacity=1.0,
            tooltip=popup_text
        ).add_to(group)

        # Plot the coverage radius
        folium.Circle(
            location=[bs.lat, bs.lon],
            radius=bs.coverage_radius_km * 1000, # Folium requires radius in meters
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity
        ).add_to(group)

    # Add groups to map
    umi_group.add_to(m)
    uma_group.add_to(m)
    rma_group.add_to(m)

    # Add Layer Control to toggle tiers on/off
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(output_file)
    print(f"✅ Preview map saved to: {to_absolute_path(output_file)}\n", flush=True)

@hydra.main(version_base=None, config_path="/Utilisateurs/dbenguer/ntn_tn_optim/configs", config_name="base")
def preview_simulation(cfg: DictConfig):
    print("\n" + "="*50)
    print("🔍 INITIALIZING DEPLOYMENT PREVIEW TOOL")
    print("="*50)
    
    # ==========================================
    # PHASE 1: THE GEOGRAPHY
    # ==========================================
    print("\n[Phase 1] Building Geographic Map...")
    with open(to_absolute_path(cfg.scenario.geojson_path)) as f:
        geometry = json.load(f)
    active_region = Region(
        name=cfg.scenario.name, 
        geojson_geometry=geometry, 
        h3_resolution=cfg.scenario.h3_resolution
    )
    tessellate_region(active_region, pad_edges=True)
    
    # ==========================================
    # PHASE 2: THE GROUND SEGMENT
    # ==========================================
    print("\n[Phase 2] Generating Ground Segment...")
    
    # 1. Spawn Users
    users = generate_users(cfg, active_region)
    print(f"✅ Generated {len(users)} Mobile Users.", flush=True)
    
    # 2. Build 5G Towers
    towers = generate_terrestrial_network(cfg, users, active_region.h3_resolution)
    
    # ==========================================
    # PHASE 3: HTML RENDER
    # ==========================================
    generate_html_map(users, towers, output_file="Deployment_Preview.html")


if __name__ == "__main__":
    preview_simulation()