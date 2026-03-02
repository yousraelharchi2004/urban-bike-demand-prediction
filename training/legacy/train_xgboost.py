"""
train_xgboost.py - XGBoost model for bike demand prediction
Target: RMSE ≤ 30
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 BIKE DEMAND PREDICTION - XGBOOST MODEL")
print("=" * 70)

# =============================================
# 1. LOAD DATA
# =============================================
print("\n📂 Loading dataset...")
df = pd.read_csv('data/raw/seoul_bike.csv', encoding='ISO-8859-1')
print(f"   Shape: {df.shape}")

# Convert date
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# =============================================
# 2. SIMPLER FEATURE ENGINEERING
# =============================================
print("\n🔧 Creating SIMPLE but MEANINGFUL features...")

# Start with basic features
features_df = df[['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 
                  'Visibility (10m)', 'Dew point temperature(°C)', 
                  'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']].copy()

# 1. SIMPLE HOUR FEATURES
features_df['hour_squared'] = features_df['Hour'] ** 2
features_df['is_peak_hour'] = ((features_df['Hour'] >= 7) & (features_df['Hour'] <= 9)) | \
                               ((features_df['Hour'] >= 17) & (features_df['Hour'] <= 19))
features_df['is_peak_hour'] = features_df['is_peak_hour'].astype(int)

# 2. SIMPLE TIME FEATURES
features_df['day_of_week'] = df['Date'].dt.dayofweek
features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
features_df['month'] = df['Date'].dt.month

# 3. SIMPLE WEATHER FEATURES
features_df['has_precipitation'] = ((features_df['Rainfall(mm)'] > 0) | (features_df['Snowfall (cm)'] > 0)).astype(int)
features_df['comfort_score'] = features_df['Temperature(°C)'] - (features_df['Humidity(%)'] / 100 * 10)
features_df['comfort_score'] = features_df['comfort_score'].clip(lower=0, upper=30)

# 4. CATEGORICAL ENCODING
# One-hot encode Seasons
seasons_dummies = pd.get_dummies(df['Seasons'], prefix='season')
features_df = pd.concat([features_df, seasons_dummies], axis=1)

# Binary encode
features_df['is_holiday'] = (df['Holiday'] == 'Holiday').astype(int)
features_df['is_functioning_day'] = (df['Functioning Day'] == 'Yes').astype(int)

print(f"✅ Created {features_df.shape[1]} SIMPLE features")

# =============================================
# 3. TARGET VARIABLE
# =============================================
y = df['Rented Bike Count'].values

# =============================================
# 4. TRAIN-TEST SPLIT (TIME-BASED)
# =============================================
print("\n🎯 Creating time-based train-test split...")

# Time-based split (80% train, 20% test)
split_idx = int(len(features_df) * 0.8)
X_train = features_df.iloc[:split_idx]
X_test = features_df.iloc[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing: {X_test.shape[0]} samples")
print(f"   Features: {X_train.shape[1]}")

# =============================================
# 5. TRAIN XGBOOST
# =============================================
print("\n⚡ Training XGBoost...")

# XGBoost parameters (tuned for regression)
xgb_model = xgb.XGBRegressor(
    n_estimators=500,           # More trees
    max_depth=6,                # Shallower to prevent overfitting
    learning_rate=0.05,         # Slower learning
    subsample=0.8,              # Prevent overfitting
    colsample_bytree=0.8,       # Prevent overfitting
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse',
    early_stopping_rounds=20
)

print("   Training model (this might take a minute)...")

# Train with validation set for early stopping
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=50  # Print every 50 iterations
)

# =============================================
# 6. EVALUATE MODEL
# =============================================
print("\n📈 Evaluating model performance...")

# Predictions
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Safe MAPE calculation
def safe_mape(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

train_mape = safe_mape(y_train, y_pred_train)
test_mape = safe_mape(y_test, y_pred_test)

print(f"\n📊 TRAINING SET:")
print(f"   RMSE: {train_rmse:.2f}")
print(f"   MAE: {train_mae:.2f}")
print(f"   R²: {train_r2:.3f}")
print(f"   MAPE: {train_mape:.1f}%")

print(f"\n📊 TEST SET:")
print(f"   RMSE: {test_rmse:.2f}")
print(f"   MAE: {test_mae:.2f}")
print(f"   R²: {test_r2:.3f}")
print(f"   MAPE: {test_mape:.1f}%")

# Compare with previous models
print(f"\n📊 COMPARISON WITH PREVIOUS MODELS:")
print(f"   1. Original Random Forest: RMSE = 112.5")
print(f"   2. Bad Improved RF: RMSE = 194.59 (data leakage)")
print(f"   3. New Overfitted RF: RMSE = 320.76 (overfitting)")
print(f"   4. XGBoost: RMSE = {test_rmse:.2f}")

if test_rmse < 112.5:
    improvement = ((112.5 - test_rmse) / 112.5) * 100
    print(f"   ✅ Improvement: {improvement:.1f}% better than original!")
else:
    degradation = ((test_rmse - 112.5) / 112.5) * 100
    print(f"   ❌ Degradation: {degradation:.1f}% worse than original")

# =============================================
# 7. FEATURE IMPORTANCE
# =============================================
print("\n🏆 Top 15 Feature Importances (XGBoost):")

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(15).iterrows():
    importance_pct = row['importance'] * 100
    print(f"   {row['feature']:25s}: {importance_pct:5.1f}%")

# =============================================
# 8. SAVE MODEL
# =============================================
print("\n💾 Saving XGBoost model...")

model_data = {
    'model': xgb_model,
    'feature_names': list(X_train.columns),
    'metrics': (test_rmse, test_mae, test_r2, test_mape),
    'training_date': pd.Timestamp.now().isoformat()
}

joblib.dump(model_data, 'models/xgboost_model.pkl')
print("✅ Model saved: models/xgboost_model.pkl")

# Save metrics
import json
metrics_dict = {
    'rmse': float(test_rmse),
    'mae': float(test_mae),
    'r2': float(test_r2),
    'mape': float(test_mape) if not np.isnan(test_mape) else None,
    'n_features': X_train.shape[1],
    'model_type': 'XGBoost',
    'parameters': str(xgb_model.get_params())
}

with open('models/xgboost_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print("✅ Metrics saved: models/xgboost_metrics.json")

# =============================================
# 9. TARGET CHECK
# =============================================
print("\n" + "=" * 70)
print("🎯 TARGET CHECK: RMSE ≤ 30")
print("=" * 70)

if test_rmse <= 30:
    print(f"🎉🎉🎉 SUCCESS! RMSE = {test_rmse:.2f} ≤ 30 🎉🎉🎉")
    print("✨ CONGRATULATIONS! Target achieved with XGBoost!")
elif test_rmse <= 50:
    print(f"👍 EXCELLENT PROGRESS! RMSE = {test_rmse:.2f} ≤ 50")
    print(f"   Need {((test_rmse - 30) / test_rmse * 100):.1f}% more improvement")
elif test_rmse <= 80:
    print(f"📈 GOOD PROGRESS! RMSE = {test_rmse:.2f} ≤ 80")
    print(f"   Need {((test_rmse - 30) / test_rmse * 100):.1f}% more improvement")
elif test_rmse <= 112.5:
    print(f"📊 BETTER THAN ORIGINAL! RMSE = {test_rmse:.2f} ≤ 112.5")
    print(f"   Need {((test_rmse - 30) / test_rmse * 100):.1f}% more improvement")
else:
    print(f"❌ STILL WORSE THAN ORIGINAL: RMSE = {test_rmse:.2f} > 112.5")

# =============================================
# 10. NEXT STEPS RECOMMENDATION
# =============================================
print("\n" + "=" * 70)
print("🔄 NEXT STEPS RECOMMENDATION")
print("=" * 70)

if test_rmse > 80:
    print("1. 🔧 Try LightGBM (often better than XGBoost)")
    print("2. 📊 Add domain-specific features:")
    print("   - Holiday calendar (day before/after holidays)")
    print("   - Weather forecasts (if available)")
    print("   - Time-series features (proper lag without leakage)")
    print("3. 🎯 Hyperparameter tuning with Bayesian Optimization")
    print("4. 🤝 Ensemble multiple models")
elif test_rmse > 50:
    print("1. 🎯 Hyperparameter tuning (GridSearchCV)")
    print("2. 📈 Add more meaningful interactions")
    print("3. 🧹 Feature selection (remove low importance)")
    print("4. 📊 Try ensemble with Random Forest")
else:
    print("1. 🎯 Final tuning to reach RMSE ≤ 30")
    print("2. 📊 Create ensemble model")
    print("3. 🚀 Deploy the model")

print("\n" + "=" * 70)
print("✅ XGBOOST TRAINING COMPLETE")
print("=" * 70)