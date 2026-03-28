import pandas as pd
from scipy.spatial import cKDTree

# Load data
weather = pd.read_csv("../data/weather_features.csv")
terrain = pd.read_csv("../data/terrain_features.csv")

# Build KDTree using terrain coordinates
tree = cKDTree(terrain[["latitude", "longitude"]].values)

# For each weather point → find nearest terrain point
dist, idx = tree.query(weather[["latitude", "longitude"]].values)

# Get matching terrain rows
matched_terrain = terrain.iloc[idx].reset_index(drop=True)

# Merge
final_df = pd.concat([weather.reset_index(drop=True), matched_terrain[["slope", "aspect"]]], axis=1)

print(final_df.head())

# Save final dataset
final_df.to_csv("../data/final_dataset.csv", index= False)