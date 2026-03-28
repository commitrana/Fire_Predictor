import rasterio
import numpy as np
import pandas as pd

#load DEM
with rasterio.open("../data/dem.tif") as src:
    dem = src.read(1)
    transform = src.transform

# Calculate gradient
dy, dx = np.gradient(dem)

# Slope (in radians → convert to degrees)
slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)

# Aspect
aspect = np.arctan2(-dx, dy) * (180 / np.pi)




# Fix aspect range
aspect = np.where(aspect < 0, 360 + aspect, aspect)

rows, cols = dem.shape

data = []

for i in range(rows):
    for j in range(cols):
        lon, lat = rasterio.transform.xy(transform, i, j)
        data.append([
            lat,
            lon,
            slope[i][j],
            aspect[i][j]
        ])

df_terrain = pd.DataFrame(data, columns=[
    "latitude", "longitude", "slope", "aspect"
])

print(df_terrain.head())

df_terrain.to_csv("../data/terrain_features.csv", index=False)