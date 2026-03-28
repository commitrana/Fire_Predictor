import pandas as pd
import joblib

# load trained model
model = joblib.load('../data/fire_model_himachal.pkl')

# load dataset
df = pd.read_csv('../data/final_dataset_himachal_scaled.csv')

# features (same as training)
features = ['temperature','humidity','wind','rain','slope','aspect','thematic']
X = df[features]

# predict probability
df['fire_risk'] = model.predict_proba(X)[:,1]

# convert to percentage
df['fire_risk_percent'] = df['fire_risk'] * 100

# categorize risk
def risk_level(p):
    if p > 70:
        return "High"
    elif p > 40:
        return "Medium"
    else:
        return "Low"

df['risk_level'] = df['fire_risk_percent'].apply(risk_level)

# save output
df.to_csv('../data/predicted_fire_risk.csv', index=False)

print("🔥 Fire risk prediction complete!")