# train_fire_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# -------------------------
# 1. Load the dataset
# -------------------------
df = pd.read_csv('../data/final_dataset_himachal_scaled.csv')

# Features and target
X = df[['temperature','humidity','wind','rain','slope','aspect','thematic']]
y = df['fire']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 2. Train Random Forest
# -------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# -------------------------
# 3. Evaluate model
# -------------------------
y_prob = rf.predict_proba(X_test)[:,1]

# lower threshold
y_pred = (y_prob > 0.3).astype(int)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# 4. Feature importance
# -------------------------
importances = rf.feature_importances_
plt.bar(X.columns, importances)
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

# -------------------------
# 5. Save the model
# -------------------------
joblib.dump(rf, '../data/fire_model_himachal.pkl')
print("Model saved as fire_model_himachal.pkl")