import pandas as pd
import json
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# 1. Models & Core Types
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.core.types import WalkerParameters, OrbitType
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation

# 2. Generators & Coverage Mapping
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region
from hybrid_ntn_optimizer.traffic.profiles import generate_users
from hybrid_ntn_optimizer.terrestrial.coverage import generate_terrestrial_network

# 3. Simulation Engine
from hybrid_ntn_optimizer.simulation.full_pipeline import run_daily_mobility_simulation
from hybrid_ntn_optimizer.visualization.plots import plot_master_hybrid_animation


print("\n" + "="*50)
print("Hydra Configuration System Initialization")
@hydra.main(version_base=None, config_path="/Utilisateurs/dbenguer/ntn_tn_optim/configs", config_name="base")
def run_simulation(cfg: DictConfig):
    print("\n" + "="*50)
    print("🚀 INITIALIZING HYBRID NTN-TN SIMULATOR (MEGA-CONSTELLATION)")
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
    # Fill the region with hexagons (including edge padding)
    tessellate_region(active_region, pad_edges=True)
    
    # ==========================================
    # PHASE 2: THE SPACE SEGMENT (MULTI-SHELL)
    # ==========================================
    print("\n[Phase 2] Generating Space Segment (Mega-Constellation)...")
    
    leos = []
    total_sats_deployed = 0
    
    # Loop through the shells defined in the YAML
    for shell_key, shell_cfg in cfg.constellation.shells.items():
        walker_params = WalkerParameters(
            total_satellites=shell_cfg.total_satellites,
            num_planes=shell_cfg.num_planes,
            phasing=shell_cfg.phasing,
            inclination_deg=shell_cfg.inclination_deg,
            altitude_km=shell_cfg.altitude_km,
            orbit_type=OrbitType.LEO
        )
        
        # Instantiate the shell using the global RF settings
        shell = LEOConstellation(
            params=walker_params,
            name=shell_cfg.name,
            eirp_dbw=cfg.constellation.get("eirp_dbw", 40.0),
            g_t_db=cfg.constellation.get("g_t_db", 10.0),
            max_spot_beams=cfg.constellation.get("max_spot_beams", 15),
            beam_radius_nadir_km=cfg.constellation.get("beam_radius_nadir_km", 200.0),
            max_steering_angle_deg=cfg.constellation.get("max_steering_angle_deg", 45.0)
        )
        leos.append(shell)
        total_sats_deployed += shell.num_satellites
        print(f"  -> Deployed {shell.name}: {shell.num_satellites} satellites at {shell_cfg.altitude_km}km.")
        
    print(f"✅ Mega-Constellation Deployed: {total_sats_deployed} total satellites across {len(leos)} shells.")
    
    # ==========================================
    # PHASE 3: THE GROUND SEGMENT
    # ==========================================
    print("\n[Phase 3] Populating Ground Segment...")
    
    # 1. Spawn Users
    users = generate_users(cfg, active_region)
    print(f"✅ Generated {len(users)} Mobile Users.", flush=True)
    
    # 2. Build 5G Towers using KMeans
    towers = generate_terrestrial_network(cfg, users, active_region.h3_resolution)
    
    # ==========================================
    # PHASE 4: THE HYBRID SIMULATION LOOP
    # ==========================================
    print("\n[Phase 4] Initiating 24-Hour Mobility & Traffic Engine...", flush=True)
    
    # Pass the LIST of constellations (leos) into the simulation engine
    beam_animation_data, user_animation_data = run_daily_mobility_simulation(
        cfg=cfg, 
        users=users, 
        base_stations=towers, 
        leos=leos, 
        region=active_region
    )
    
    # ==========================================
    # PHASE 5: THE MASTER VISUALIZATION
    # ==========================================
    plot_master_hybrid_animation(
        region=active_region, 
        users=users, 
        base_stations=towers, 
        beam_data=beam_animation_data, 
        user_data=user_animation_data,
        duration_s=cfg.simulation.duration_s, 
        time_step_s=cfg.simulation.time_step_s,
        filename="Final_Animation.html"
    )
    
    print("\n🎉 SIMULATION COMPLETE. Check output directories for CSV exports.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_simulation()