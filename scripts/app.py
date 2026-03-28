import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("fire_model_himachal.pkl")

st.set_page_config(page_title="🔥 Fire Predictor", layout="centered")

st.title("🔥 Himachal Forest Fire Prediction System")

# ==========================
# INPUT MODE TOGGLE
# ==========================
input_mode = st.radio("Select Input Mode:", ["Slider Input", "Manual Input"])

st.markdown("### 🌍 Enter Parameters")

# ==========================
# INPUTS
# ==========================
col1, col2 = st.columns(2)

if input_mode == "Slider Input":

    with col1:
        latitude = st.slider("Latitude", 30.0, 35.0, 31.5)
        temperature = st.slider("Temperature (Kelvin)", 270, 320, 295)
        wind = st.slider("Wind Speed (m/s)", 0.0, 15.0, 5.0)
        slope = st.slider("Slope (degrees)", 0.0, 180.0, 10.0)

    with col2:
        longitude = st.slider("Longitude", 75.0, 80.0, 77.0)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
        rain = st.slider("Rainfall (mm)", 0.0, 20.0, 2.0)
        aspect = st.slider("Aspect (0–360°)", 0.0, 360.0, 180.0)

else:  # Manual Input

    with col1:
        latitude = st.number_input("Latitude", value=31.5)
        temperature = st.number_input("Temperature (Kelvin)", value=295.0)
        wind = st.number_input("Wind Speed (m/s)", value=5.0)
        slope = st.number_input("Slope (degrees)", min_value=0.0, max_value=180.0, value=10.0)

    with col2:
        longitude = st.number_input("Longitude", value=77.0)
        humidity = st.number_input("Humidity (%)", value=50.0)
        rain = st.number_input("Rainfall (mm)", value=2.0)
        aspect = st.number_input("Aspect (0–360°)", min_value=0.0, max_value=360.0, value=180.0)

thematic = st.selectbox("Land Cover Type (Thematic)", [0, 10, 20, 30, 40, 50])

# ==========================
# PREDICTION
# ==========================
if st.button("Predict 🔥"):

    input_data = np.array([[
        latitude,
        longitude,
        temperature,
        humidity,
        wind,
        rain,
        slope,
        aspect,
        thematic
    ]])

    prob = model.predict_proba(input_data)[0][1] * 100

    st.subheader(f"🔥 Fire Probability: {prob:.2f}%")

    if prob < 30:
        st.success("✅ Low Risk")
    elif prob < 70:
        st.warning("⚠️ Moderate Risk")
    else:
        st.error("🔥 High Risk")

# ==========================
# FEATURE IMPORTANCE
# ==========================
st.markdown("---")
st.subheader("📊 Feature Importance")

features = [
    'latitude','longitude','temperature',
    'humidity','wind','rain','slope','aspect','thematic'
]

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots()
ax.bar(range(len(features)), importances[indices])
ax.set_xticks(range(len(features)))
ax.set_xticklabels(np.array(features)[indices], rotation=45)

for i, v in enumerate(importances[indices]):
    ax.text(i, v + 0.005, f"{v:.2f}", ha='center')

ax.set_title("Feature Importance")
ax.set_ylabel("Importance Score")

st.pyplot(fig)

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("Random Forest • Spatial + Environmental Fire Prediction")