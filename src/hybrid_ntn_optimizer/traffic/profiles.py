import numpy as np
import random
import math
from typing import List
from omegaconf import DictConfig, OmegaConf
import rasterio
import concurrent.futures
import multiprocessing
from shapely.geometry import shape, Point  # Enforces spatial masking
from hybrid_ntn_optimizer.core.utils import _detect_cpus
from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.scenario import Region
import os, pickle

def _dump_users(users, path="data/users.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(users, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"💾 Saved {len(users):,} users (full objects) to {path}")
    
    
# Module-level globals, populated once per worker by the pool initializer.
_BP = None        # boundary polygon (shapely)
_CFG = None       # config
_H3_RES = None    # h3 resolution


def generate_users_WITHOUT_TIF(cfg: DictConfig, region: Region) -> List[User]:
    print("Generating Mobile Subscriber Population using Repository Mesh Grid Masking...")
    users = []
    user_id_counter = 0
    
    num_city = cfg.population.total_city_users
    num_rural = cfg.population.total_rural_users
    
    cities_dict = cfg.population.cities
    city_coords = [list(c.coords) for c in cities_dict.values()]
    city_weights = [c.weight for c in cities_dict.values()]
    
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    
    # 1. Parse the native GeoJSON geometry from the region into a Shapely polygon
    if hasattr(region, 'geojson_geometry') and region.geojson_geometry:
        if not isinstance(region.geojson_geometry, dict):
            boundary_dict = OmegaConf.to_container(region.geojson_geometry, resolve=True)
        else:
            boundary_dict = region.geojson_geometry
        boundary_polygon = shape(boundary_dict)
    else:
        raise ValueError("❌ region.geojson_geometry is missing or empty! Cannot enforce strict repository masking.")
        
    # Get the exact bounding box of Ontario from the geometry (minx, miny, maxx, maxy)
    min_lon, min_lat, max_lon, max_lat = boundary_polygon.bounds

    # ==========================================
    # 2. GENERATE URBAN USERS (Repository Approach)
    # ==========================================
    for _ in range(num_city):
        is_inside = False
        lat, lon = 0.0, 0.0
        
        while not is_inside:
            center_idx = np.random.choice(len(city_coords), p=city_weights)
            center = city_coords[center_idx]
            # Gaussian distribution centered on the exact city coordinates
            lat = np.random.normal(center[0], cfg.population.city_scatter_std_dev)
            lon = np.random.normal(center[1], cfg.population.city_scatter_std_dev)
            
            # Strict spatial containment check
            if boundary_polygon.contains(Point(lon, lat)):
                is_inside = True
                
        users.append(_build_user_profile(user_id_counter, lat, lon, region.h3_resolution, cfg, boundary_polygon))
        user_id_counter += 1

    # ==========================================
    # 3. GENERATE RURAL USERS (Repository Approach)
    # ==========================================
    for _ in range(num_rural):
        is_inside = False
        lat, lon = 0.0, 0.0
        
        while not is_inside:
            # Uniform random sampling across the true latitude/longitude span of Ontario
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            
            # Reject any coordinate falling outside provincial borders
            if boundary_polygon.contains(Point(lon, lat)):
                is_inside = True
                
        users.append(_build_user_profile(user_id_counter, lat, lon, region.h3_resolution, cfg, boundary_polygon))
        user_id_counter += 1
        
    return users


def _parallel_user_worker(args):
    uid, base_lat, base_lon = args          # tasks are now tiny

    # Per-user reseed so each user gets independent jitter.
    np.random.seed(uid)
    random.seed(uid)

    lat = base_lat + np.random.uniform(-0.0004, 0.0004)
    lon = base_lon + np.random.uniform(-0.0004, 0.0004)

    if not _BP.contains(Point(lon, lat)):
        lat, lon = base_lat, base_lon

    return _build_user_profile(uid, lat, lon, _H3_RES, _CFG, _BP)


def generate_users(cfg: DictConfig, region: Region) -> List[User]:
    users = []
    user_id_counter = 0
    
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    
    # 1. Parse the native GeoJSON geometry from the region into a Shapely polygon
    # (Both generators need this to enforce borders)
    if hasattr(region, 'geojson_geometry') and region.geojson_geometry:
        if not isinstance(region.geojson_geometry, dict):
            boundary_dict = OmegaConf.to_container(region.geojson_geometry, resolve=True)
        else:
            boundary_dict = region.geojson_geometry
        boundary_polygon = shape(boundary_dict)
    else:
        raise ValueError("❌ region.geojson_geometry is missing or empty! Cannot enforce strict repository masking.")
        
    # Get the exact bounding box of Ontario from the geometry (minx, miny, maxx, maxy)
    min_lon, min_lat, max_lon, max_lat = boundary_polygon.bounds

    # Check the config for the toggle boolean (defaults to False if you don't add it)
    use_worldpop = cfg.population.get("use_worldpop", False)

    # ==========================================
    # ENGINE A: ORIGINAL REPOSITORY MESH GRID 
    # ==========================================
    if not use_worldpop:
        print("Generating Mobile Subscriber Population using Repository Mesh Grid Masking...")
        
        num_city = cfg.population.total_city_users
        num_rural = cfg.population.total_rural_users
        
        cities_dict = cfg.population.cities
        city_coords = [list(c.coords) for c in cities_dict.values()]
        city_weights = [c.weight for c in cities_dict.values()]
        
        # GENERATE URBAN USERS
        for _ in range(num_city):
            is_inside = False
            lat, lon = 0.0, 0.0
            
            while not is_inside:
                center_idx = np.random.choice(len(city_coords), p=city_weights)
                center = city_coords[center_idx]
                lat = np.random.normal(center[0], cfg.population.city_scatter_std_dev)
                lon = np.random.normal(center[1], cfg.population.city_scatter_std_dev)
                
                if boundary_polygon.contains(Point(lon, lat)):
                    is_inside = True
                    
            users.append(_build_user_profile(user_id_counter, lat, lon, region.h3_resolution, cfg, boundary_polygon))
            user_id_counter += 1

        # GENERATE RURAL USERS
        for _ in range(num_rural):
            is_inside = False
            lat, lon = 0.0, 0.0
            
            while not is_inside:
                lat = np.random.uniform(min_lat, max_lat)
                lon = np.random.uniform(min_lon, max_lon)
                
                if boundary_polygon.contains(Point(lon, lat)):
                    is_inside = True
                    
            users.append(_build_user_profile(user_id_counter, lat, lon, region.h3_resolution, cfg, boundary_polygon))
            user_id_counter += 1

    # ==========================================
    # ENGINE B: WORLDPOP RASTER SAMPLING
    # ==========================================
    else:
        print("Generating Mobile Subscriber Population using WorldPop Raster Data...")
        
        # Combine city and rural counts for the total WorldPop pool
        num_users = cfg.population.total_city_users + cfg.population.total_rural_users
        tif_path = cfg.population.get("worldpop_tif_path", "data/can_ppp_2020_UNadj.tif")
        
        print(f"🌍 Loading WorldPop demographic data from: {tif_path}")
        
        with rasterio.open(tif_path) as dataset:
            row_min, col_min = dataset.index(min_lon, max_lat)
            row_max, col_max = dataset.index(max_lon, min_lat)
            
            row_min, row_max = max(0, row_min), min(dataset.height, row_max)
            col_min, col_max = max(0, col_min), min(dataset.width, col_max)
            
            region_pop_data = dataset.read(1)[row_min:row_max, col_min:col_max]
            nodata = dataset.nodata
            
            if nodata is not None:
                valid_mask = (region_pop_data != nodata) & (region_pop_data > 0)
            else:
                valid_mask = region_pop_data > 0
                
            valid_rows, valid_cols = np.where(valid_mask)
            valid_pops = region_pop_data[valid_mask]
            
            global_rows = valid_rows + row_min
            global_cols = valid_cols + col_min
            
            lons, lats = rasterio.transform.xy(dataset.transform, global_rows, global_cols, offset='center')
            
            print("✂️ Masking population data to exact Region boundaries...")
            strict_lons, strict_lats, strict_pops = [], [], []
            
            for i in range(len(lons)):
                if boundary_polygon.contains(Point(lons[i], lats[i])):
                    strict_lons.append(lons[i])
                    strict_lats.append(lats[i])
                    strict_pops.append(valid_pops[i])

            if cfg.population.get("worldpop_tif_restricted", None):        
                if len(strict_pops) == 0:
                    raise ValueError("No population found inside the provided polygon!")
                    
                strict_pops = np.array(strict_pops)
                probabilities = strict_pops / strict_pops.sum()
                
                print(f"📊 Sampling {num_users} users across {len(strict_pops)} populated pixels...")
                sampled_indices = np.random.choice(len(strict_pops), size=num_users, p=probabilities)
                
                for idx in sampled_indices:
                    lat = strict_lats[idx] + np.random.uniform(-0.0004, 0.0004)
                    lon = strict_lons[idx] + np.random.uniform(-0.0004, 0.0004)
                    
                    if not boundary_polygon.contains(Point(lon, lat)):
                        lat, lon = strict_lats[idx], strict_lons[idx]
                        
                    users.append(_build_user_profile(user_id_counter, lat, lon, region.h3_resolution, cfg, boundary_polygon))
                    user_id_counter += 1
            else:
                if len(strict_pops) == 0:
                    raise ValueError("No population found inside the provided polygon!")
                strict_pops = np.array(strict_pops)
                total_expected_users = int(np.round(strict_pops).sum())
                print(f"📊 WorldPop Scan Complete: Found EXACTLY {total_expected_users:,} humans in this region.")
            
                print("⚙️ Preparing task batches for CPU cores...")
                tasks = []
                user_id_counter = 0
                
                print("📊 Spawning users based EXACTLY on WorldPop census counts...")
                
                # --- EXHAUSTIVE SPAWNING ---
                for idx in range(len(strict_pops)):
                    people_in_this_pixel = int(np.round(strict_pops[idx]))
                    for _ in range(people_in_this_pixel):
                        # We pack the arguments into a tuple for the worker
                        tasks.append((user_id_counter, strict_lats[idx], strict_lons[idx]))
                        user_id_counter += 1
                global _BP, _CFG, _H3_RES          
                _BP, _CFG, _H3_RES = boundary_polygon, cfg, region.h3_resolution
                num_cores = max(1, _detect_cpus() - 1)  # Leave one core free for the OS
                print(f"🚀 Firing up {num_cores} CPU cores for parallel generation. This may take a few minutes...")
                # Chunksize batches the users together so the CPU cores aren't constantly communicating
                chunk_size = max(1, len(tasks) // (num_cores * 4))  
                print(f"Number of tasks is {len(tasks):,}. ⚡ Using a chunk size of {chunk_size} for parallel processing.") 
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
                    # We use a loop here instead of list() so we can print progress updates!
                    for spawned_user in executor.map(_parallel_user_worker, tasks, chunksize=chunk_size):
                        users.append(spawned_user)
                            
        print("✅ WorldPop generation complete!", flush=True)
    _dump_users(users, path="/Utilisateurs/dbenguer/ntn_tn_optim/data/users.pkl")
    return users

def _build_user_profile(uid: int, lat: float, lon: float, res: int, cfg: DictConfig, boundary_polygon) -> User:
    roll = np.random.rand()
    cumulative_prob = 0.0
    u_type, demand = "Unknown", 0.0
    
    for profile_name, profile_data in cfg.population.traffic.profiles.items():
        cumulative_prob += profile_data.probability
        if roll <= cumulative_prob:
            u_type = str(profile_name).capitalize() 
            demand = np.random.uniform(profile_data.min_mbps, profile_data.max_mbps)
            break
            
    if u_type == "Unknown":
        fallback_name = list(cfg.population.traffic.profiles.keys())[-1]
        fallback_data = cfg.population.traffic.profiles[fallback_name]
        u_type = str(fallback_name).capitalize()
        demand = np.random.uniform(fallback_data.min_mbps, fallback_data.max_mbps)

    diurnal_dict = OmegaConf.to_container(cfg.population.traffic.diurnal_curve, resolve=True)
    mobility_dict = OmegaConf.to_container(cfg.population.mobility, resolve=True)
        
    user = User(
        user_id=uid, home_lat=lat, home_lon=lon, user_type=u_type, 
        base_demand_mbps=demand, diurnal_cfg=diurnal_dict, mobility_cfg=mobility_dict
    )
    user.set_resolution(res)
    
    # Configure human mobility dynamics indices
    num_attractors = cfg.population.mobility.num_attractors
    ranks = np.arange(1, num_attractors + 1)
    raw_probs = 1.0 / (ranks ** cfg.population.mobility.zipf_alpha)
    user.attractor_probs = raw_probs / np.sum(raw_probs)
    
    # ==========================================
    # 4. PROTECTED ATTRACTOR GENERATION 
    # ==========================================
    user.attractors = [(lat, lon)]
    for _ in range(num_attractors - 1):
        accepted_destination = False
        attractor_lat, attractor_lon = 0.0, 0.0
        
        # Enforce that no daily movement path vectors step across the boundary
        attempt_counter = 0
        while not accepted_destination:
            accepted = False
            r_km = 0.0
            while not accepted:
                r_km = np.random.pareto(cfg.population.mobility.pareto_beta - 1.0) * cfg.population.mobility.delta_r0_km
                if np.random.rand() < np.exp(-r_km / cfg.population.mobility.cutoff_kappa_km):
                    accepted = True
            
            earth_radius_km = 6371.0
            r_deg = math.degrees(r_km / earth_radius_km)
            theta = np.random.uniform(0, 2 * np.pi)
            
            attractor_lat = lat + (r_deg * math.degrees(math.sin(theta)))
            attractor_lon = lon + (r_deg * math.degrees(math.cos(theta)) / math.cos(math.radians(lat)))
            
            if boundary_polygon.contains(Point(attractor_lon, attractor_lat)):
                accepted_destination = True
                
        user.attractors.append((attractor_lat, attractor_lon))
        
    return user