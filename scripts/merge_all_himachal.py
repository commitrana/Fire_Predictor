import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from scipy.interpolate import griddata


# -------------------------
# 1. Create dense Himachal grid
# -------------------------
lat_min, lat_max = 30.0, 33.5
lon_min, lon_max = 75.6, 79.1
lat_step, lon_step = 0.01, 0.01
lats = np.arange(lat_min, lat_max + lat_step, lat_step)
lons = np.arange(lon_min, lon_max + lon_step, lon_step)
lat_grid, lon_grid = np.meshgrid(lats, lons)
lat_flat = lat_grid.ravel()
lon_flat = lon_grid.ravel()



# -------------------------
# 2. Load terrain data
# -------------------------
terrain = pd.read_csv('../data/terrain_features.csv')
tree_terrain = cKDTree(np.column_stack([terrain['latitude'], terrain['longitude']]))
_, idx = tree_terrain.query(np.column_stack([lat_flat, lon_flat]))
slope_flat = terrain['slope'].values[idx]
aspect_flat = terrain['aspect'].values[idx]

# -------------------------
# 3. Load thematic raster
# -------------------------
thematic_raster = "../data/thematic.tif"  # path to your thematic.tif file
with rasterio.open(thematic_raster) as src:

    thematic_arr = src.read(1)
    # Convert lat/lon to row/col indices
    # Stack lon/lat pairs
    coords = np.column_stack([lon_flat, lat_flat])

    # Sample raster values safely
    thematic_flat = np.array([val[0] for val in src.sample(coords)])

# -------------------------
# 4. Map weather data using nearest neighbor
# -------------------------

weather_df = pd.read_csv('../data/weather_features.csv')
tree_weather = cKDTree(weather_df[['latitude','longitude']].values)

# Query nearest weather point for each grid cell
_, idx = tree_weather.query(np.column_stack([lat_flat, lon_flat]))

# Assign values
temp_flat = weather_df['temperature'].values[idx]
humidity_flat = weather_df['humidity'].values[idx]
wind_flat = weather_df['wind_speed'].values[idx]
rain_flat = weather_df['rainfall'].values[idx]

# -------------------------
# 5. Map fire data
# -------------------------
fire_df = pd.read_csv('../data/fire_filtered.csv')
tree_fire = cKDTree(fire_df[['latitude','longitude']].values)
distances, idx = tree_fire.query(np.column_stack([lat_flat, lon_flat]), distance_upper_bound=0.02)  # 0.02 deg ~2 km
fire_flat = np.zeros_like(lat_flat)
fire_flat[distances != np.inf] = 1  # mark 1 if fire nearby

# -------------------------
# 6. Create final dataframe
# -------------------------
final_df = pd.DataFrame({
    "latitude": lat_flat,
    "longitude": lon_flat,
    "temperature": temp_flat,
    "humidity": humidity_flat,
    "wind": wind_flat,
    "rain": rain_flat,
    "slope": slope_flat,
    "aspect": aspect_flat,
    "thematic": thematic_flat,
    "fire": fire_flat
})

# -------------------------
# 7. Save CSV
# -------------------------
final_df.to_csv('../data/final_dataset_himachal_scaled.csv', index=False)
