import pandas as pd

# Load fire data
df = pd.read_csv("../data/fire_data.csv")

# Filter your region
df_filtered = df[
    (df["latitude"] >= 30) & (df["latitude"] <= 31) &
    (df["longitude"] >= 77) & (df["longitude"] <= 78)
]

print(df_filtered.shape)

# Save filtered data
df_filtered.to_csv("../data/fire_filtered.csv", index=False)