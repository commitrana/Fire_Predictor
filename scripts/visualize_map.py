import pandas as pd
import folium
from folium.plugins import HeatMap

# load predicted data
df = pd.read_csv('../data/predicted_fire_risk.csv')

# create base map (Himachal center approx)
m = folium.Map(location=[31.1048, 77.1734], zoom_start=7)

# separate high and low risk
high_risk = df[df['fire_risk'] == 1]
low_risk = df[df['fire_risk'] == 0]

# smart sampling (no crash)
high_sample = high_risk.sample(min(len(high_risk), 800), random_state=42)
low_sample = low_risk.sample(min(len(low_risk), 200), random_state=42)

# combine
sample_df = pd.concat([high_sample, low_sample])

# 🔥 weighted heatmap data (IMPORTANT FIX)
heat_data = [
    [row['latitude'], row['longitude'], row['fire_risk_percent']]
    for _, row in sample_df.iterrows()
]

# add heatmap (ONLY ONCE)
HeatMap(heat_data, radius=10, blur=15, max_zoom=10).add_to(m)

# function for color (FIXED)
def get_color(risk):
    risk = str(risk).lower()
    if risk == "high":
        return "red"
    elif risk == "medium":
        return "orange"
    else:
        return "green"

# add points (USE SAMPLE, NOT FULL DATASET ❗)
for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4 if str(row['risk_level']).lower() == 'high' else 2,
        color=get_color(row['risk_level']),
        fill=True,
        fill_opacity=0.6,
        popup=f"Risk: {row['fire_risk_percent']:.2f}%"
    ).add_to(m)

# save map
m.save('../data/fire_risk_map.html')

print("🔥 Map created successfully!")