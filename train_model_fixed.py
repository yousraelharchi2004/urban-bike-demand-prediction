"""
train_model_fixed.py - Fixed feature engineering without data leakage
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("🚀 BIKE DEMAND PREDICTION - FIXED FEATURE ENGINEERING")
print("=" * 60)

# =============================================
# 1. LOAD AND PREPARE DATA
# =============================================
print("\n📂 Loading dataset...")
data_path = 'data/raw/seoul_bike.csv'
df = pd.read_csv(data_path, encoding='unicode_escape')

print(f"   Original shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Rename columns for consistency
df.columns = [col.strip() for col in df.columns]

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# =============================================
# 2. BASIC FEATURE ENGINEERING (NO DATA LEAKAGE)
# =============================================
print("\n🔧 Creating safe features (NO data leakage)...")

# Create date/time features
df['day_of_week'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)

# Hour-based features (SAFE)
df['hour_squared'] = df['Hour'] ** 2
df['hour_cubed'] = df['Hour'] ** 3

# Time period flags
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_peak_hour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)) | ((df['Hour'] >= 17) & (df['Hour'] <= 19))
df['is_peak_hour'] = df['is_peak_hour'].astype(int)

# Rush hour intensity
def rush_hour_intensity(hour):
    if hour in [7, 8, 9, 17, 18, 19]:
        return 1.0
    elif hour in [6, 10, 16, 20]:
        return 0.5
    else:
        return 0.0

df['rush_hour_intensity'] = df['Hour'].apply(rush_hour_intensity)

# Hour bins (NEW - IMPORTANT!)
df['hour_bin_morning'] = ((df['Hour'] >= 6) & (df['Hour'] <= 9)).astype(int)  # 6-9 AM
df['hour_bin_afternoon'] = ((df['Hour'] >= 12) & (df['Hour'] <= 14)).astype(int)  # 12-2 PM
df['hour_bin_evening'] = ((df['Hour'] >= 17) & (df['Hour'] <= 20)).astype(int)  # 5-8 PM
df['hour_bin_night'] = ((df['Hour'] >= 21) | (df['Hour'] <= 5)).astype(int)  # 9 PM-5 AM

# Interaction terms (SAFE)
df['temp_hour_interaction'] = df['Temperature(°C)'] * df['Hour']
df['temp_humidity_interaction'] = df['Temperature(°C)'] * df['Humidity(%)']

# Weather comfort features
df['apparent_temperature'] = df['Temperature(°C)'] + 0.33 * df['Humidity(%)'] / 100 * 6.105 * np.exp(17.27 * df['Temperature(°C)'] / (237.7 + df['Temperature(°C)']))

# Weather severity score (NO FUTURE DATA)
def calculate_weather_severity(row):
    score = 0
    
    # Temperature severity
    temp = row['Temperature(°C)']
    if temp < 0 or temp > 35:
        score += 3
    elif temp < 5 or temp > 30:
        score += 2
    elif temp < 10 or temp > 25:
        score += 1
    
    # Precipitation severity
    if row['Rainfall(mm)'] > 5:
        score += 3
    elif row['Rainfall(mm)'] > 1:
        score += 2
    elif row['Rainfall(mm)'] > 0:
        score += 1
    
    if row['Snowfall (cm)'] > 5:
        score += 3
    elif row['Snowfall (cm)'] > 1:
        score += 2
    elif row['Snowfall (cm)'] > 0:
        score += 1
    
    # Wind severity
    if row['Wind speed (m/s)'] > 10:
        score += 3
    elif row['Wind speed (m/s)'] > 5:
        score += 2
    elif row['Wind speed (m/s)'] > 3:
        score += 1
    
    return score

df['weather_severity_score'] = df.apply(calculate_weather_severity, axis=1)
df['comfort_index'] = 10 - df['weather_severity_score']
df['comfort_index'] = df['comfort_index'].clip(lower=0, upper=10)

# Temperature bins (NEW - IMPORTANT!)
df['temp_bin_very_cold'] = (df['Temperature(°C)'] < 0).astype(int)
df['temp_bin_cold'] = ((df['Temperature(°C)'] >= 0) & (df['Temperature(°C)'] < 10)).astype(int)
df['temp_bin_cool'] = ((df['Temperature(°C)'] >= 10) & (df['Temperature(°C)'] < 15)).astype(int)
df['temp_bin_comfortable'] = ((df['Temperature(°C)'] >= 15) & (df['Temperature(°C)'] <= 25)).astype(int)
df['temp_bin_warm'] = ((df['Temperature(°C)'] > 25) & (df['Temperature(°C)'] <= 30)).astype(int)
df['temp_bin_hot'] = (df['Temperature(°C)'] > 30).astype(int)

# Visibility categories
df['visibility_good'] = (df['Visibility (10m)'] > 1500).astype(int)
df['visibility_poor'] = (df['Visibility (10m)'] <= 1000).astype(int)

# Season-specific interactions
df['summer_temp_effect'] = df['Temperature(°C)'] * (df['Seasons'] == 'Summer').astype(int)
df['winter_temp_effect'] = df['Temperature(°C)'] * (df['Seasons'] == 'Winter').astype(int)

# =============================================
# 3. ENCODE CATEGORICAL VARIABLES
# =============================================
print("\n🔤 Encoding categorical features...")

# One-hot encode Seasons
seasons_dummies = pd.get_dummies(df['Seasons'], prefix='Seasons')
df = pd.concat([df, seasons_dummies], axis=1)

# Binary encode Holiday and Functioning Day
df['Holiday_No Holiday'] = (df['Holiday'] == 'No Holiday').astype(int)
df['Functioning Day_Yes'] = (df['Functioning Day'] == 'Yes').astype(int)

# Drop original categorical columns
df = df.drop(['Date', 'Seasons', 'Holiday', 'Functioning Day'], axis=1)

# =============================================
# 4. PREPARE TRAINING DATA
# =============================================
print("\n📊 Preparing final dataset...")

# Target variable
y = df['Rented Bike Count'].values

# Features (EXCLUDE TARGET AND ANY DATE-RELATED THAT MIGHT CAUSE LEAKAGE)
exclude_cols = ['Rented Bike Count']
X = df.drop(exclude_cols, axis=1)

print(f"✅ Final feature set: {X.shape[1]} features")
print(f"✅ Dataset shape: X={X.shape}, y={y.shape}")
print(f"\n📋 Features used: {list(X.columns)}")

# =============================================
# 5. TRAIN-TEST SPLIT (TIME-BASED)
# =============================================
print("\n🎯 Creating train-test split...")

# Use time-based split (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# =============================================
# 6. TRAIN RANDOM FOREST MODEL
# =============================================
print("\n🌲 Training Random Forest model...")

# Simplified parameter grid (focus on preventing overfitting)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Create and train model
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

print("   Performing grid search...")
grid_search = GridSearchCV(
    rf_model, 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

print(f"\n✅ Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

# =============================================
# 7. EVALUATE MODEL
# =============================================
print("\n📈 Evaluating model performance...")

# Predictions
y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Calculate MAPE (careful with zero values)
def safe_mape(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

train_mape = safe_mape(y_train, y_pred_train)
test_mape = safe_mape(y_test, y_pred_test)

print(f"\n📊 TRAINING SET PERFORMANCE:")
print(f"   RMSE: {train_rmse:.2f}")
print(f"   MAE: {train_mae:.2f}")
print(f"   R² Score: {train_r2:.3f}")
print(f"   MAPE: {train_mape:.1f}%")

print(f"\n📊 TEST SET PERFORMANCE:")
print(f"   RMSE: {test_rmse:.2f}")
print(f"   MAE: {test_mae:.2f}")
print(f"   R² Score: {test_r2:.3f}")
print(f"   MAPE: {test_mape:.1f}%")

# Compare with baseline
baseline_rmse = 112.5
improvement = ((baseline_rmse - test_rmse) / baseline_rmse) * 100

print(f"\n📊 COMPARISON WITH BASELINE (RMSE=112.5):")
print(f"   New RMSE: {test_rmse:.2f}")
print(f"   Improvement: {improvement:.1f}%")
if test_rmse <= 30:
    print(f"   🎉 TARGET ACHIEVED! RMSE ≤ 30")
elif test_rmse <= 50:
    print(f"   👍 Good progress! RMSE ≤ 50")
elif test_rmse <= 80:
    print(f"   📈 Getting better! RMSE ≤ 80")
else:
    print(f"   ⚠️  Still needs work. RMSE > 80")

# =============================================
# 8. FEATURE IMPORTANCE ANALYSIS
# =============================================
print("\n🏆 Top 15 Most Important Features:")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

# Display top 15 features
for i, row in feature_importance.head(15).iterrows():
    importance_pct = row['importance'] * 100
    print(f"   {row['feature']:30s}: {importance_pct:5.1f}%")

# Visualize feature importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance_fixed.png', dpi=100, bbox_inches='tight')
print("\n📊 Feature importance plot saved: 'feature_importance_fixed.png'")

# =============================================
# 9. SAVE MODEL AND RESULTS
# =============================================
print("\n💾 Saving model and metrics...")

# Save model
model_path = 'models/fixed_random_forest.pkl'
joblib.dump(best_rf, model_path)
print(f"✅ Model saved to: {model_path}")

# Save feature names
feature_names_path = 'models/fixed_model_features.pkl'
joblib.dump(list(X.columns), feature_names_path)
print(f"✅ Feature names saved to: {feature_names_path}")

# Save metrics
metrics = {
    'test_rmse': float(test_rmse),
    'test_mae': float(test_mae),
    'test_r2': float(test_r2),
    'test_mape': float(test_mape) if not np.isnan(test_mape) else None,
    'train_rmse': float(train_rmse),
    'train_mae': float(train_mae),
    'train_r2': float(train_r2),
    'baseline_rmse': float(baseline_rmse),
    'improvement_percent': float(improvement),
    'n_features': int(X.shape[1]),
    'best_params': grid_search.best_params_,
    'feature_importances': feature_importance.to_dict('records')
}

import json
metrics_path = 'models/fixed_model_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Metrics saved to: {metrics_path}")

# =============================================
# 10. SUMMARY
# =============================================
print("\n" + "=" * 60)
print("📋 PROJECT SUMMARY")
print("=" * 60)
print(f"Model: Random Forest Regressor (Fixed)")
print(f"Features: {X.shape[1]} engineered features")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Test R²: {test_r2:.3f}")
print(f"Target RMSE ≤ 30: {'✅ ACHIEVED' if test_rmse <= 30 else '❌ NOT ACHIEVED'}")
print(f"Improvement from baseline: {improvement:.1f}%")

if test_rmse > 30:
    print("\n" + "=" * 60)
    print("🔄 NEXT STEPS:")
    print("=" * 60)
    print("1. Try XGBoost or LightGBM (usually better than Random Forest)")
    print("2. Add more domain-specific features")
    print("3. Try time-series models (ARIMA, Prophet)")
    print("4. Ensemble multiple models")
    print("5. More extensive hyperparameter tuning")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)