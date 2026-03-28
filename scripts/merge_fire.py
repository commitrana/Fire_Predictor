import pandas as pd
from scipy.spatial import cKDTree

# Load data
df = pd.read_csv("../data/final_dataset.csv")
fire = pd.read_csv("../data/fire_filtered.csv")

# Build tree
tree = cKDTree(df[["latitude", "longitude"]].values)

# Query
dist, idx = tree.query(fire[["latitude", "longitude"]].values)

# Initialize
df["fire"] = 0

# UNIQUE indices only
unique_idx = list(set(idx))

# Limit number of fire points (VERY IMPORTANT)
top_n = min(5, len(unique_idx))

selected_idx = unique_idx[:top_n]

# Mark fire
df.loc[selected_idx, "fire"] = 1

print(df["fire"].value_counts())

# Save
df.to_csv("../data/final_dataset_with_fire.csv", index=False)