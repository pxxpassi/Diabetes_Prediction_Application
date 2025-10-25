# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load exported data
df = pd.read_csv("processed_diabetes_data.csv")

# Select the relevant features (adjust if your CSV columns differ)
X = df[['pregnancies', 'glucose_imp', 'blood_pressure_imp', 'skin_thickness_imp',
        'insulin_imp', 'bmi_imp', 'diabetes_pedigree', 'age', 'diabetes_risk_score']]
y = df['diabetes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Save model
joblib.dump(rf, "diabetes_model.pkl")

print("✅ Model trained and saved as diabetes_model.pkl")
