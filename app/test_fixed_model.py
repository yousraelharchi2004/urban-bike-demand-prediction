"""
test_fixed_model.py - Test the fixed model
"""

import joblib
import pandas as pd
import numpy as np

# Load the fixed model
print("Loading fixed model...")
model = joblib.load('models/fixed_random_forest.pkl')
features = joblib.load('models/fixed_model_features.pkl')

print(f"Model loaded with {len(features)} features")

# Create a test sample (similar to your frontend input)
test_data = {
    'Hour': 16,
    'Temperature(°C)': 22.5,
    'Humidity(%)': 60,
    'Wind speed (m/s)': 2.5,
    'Visibility (10m)': 2000,
    'Dew point temperature(°C)': 10,
    'Solar Radiation (MJ/m2)': 1.5,
    'Rainfall(mm)': 0,
    'Snowfall (cm)': 0,
    'day_of_week': 2,  # Wednesday
    'month': 6,  # June
    'year': 2018,
    'is_month_start': 0,
    'is_month_end': 0,
    'Seasons_Spring': 0,
    'Seasons_Summer': 1,
    'Seasons_Winter': 0,
    'Holiday_No Holiday': 1,
    'Functioning Day_Yes': 1
}

# Create DataFrame with all expected features
test_df = pd.DataFrame([test_data])

# Make sure we have all the features the model expects
for feature in features:
    if feature not in test_df.columns:
        test_df[feature] = 0  # Add missing features with default value

# Reorder columns to match model expectation
test_df = test_df[features]

# Make prediction
prediction = model.predict(test_df)
print(f"\n🎯 Prediction: {prediction[0]:.0f} bikes")