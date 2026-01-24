"""
emergency_diagnostics_simple.py - Find out why models are failing catastrophically
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚨🚨🚨 EMERGENCY DATA DIAGNOSTICS 🚨🚨🚨")
print("=" * 80)

# =============================================
# 1. LOAD AND INSPECT DATA
# =============================================
print("\n📂 1. LOADING AND INSPECTING DATA")
print("-" * 40)

df = pd.read_csv('data/raw/seoul_bike.csv', encoding='ISO-8859-1')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# Check first few rows
print("\nFirst 3 rows:")
print(df.head(3))

# Check data types
print("\nData types:")
print(df.dtypes)

# =============================================
# 2. TARGET VARIABLE ANALYSIS
# =============================================
print("\n\n📊 2. TARGET VARIABLE ANALYSIS ('Rented Bike Count')")
print("-" * 40)

target = df['Rented Bike Count']
print(f"Mean: {target.mean():.1f}")
print(f"Std: {target.std():.1f}")
print(f"Min: {target.min()}")
print(f"Max: {target.max()}")
print(f"Median: {target.median():.1f}")
print(f"Zeros: {(target == 0).sum()} ({(target == 0).sum()/len(target)*100:.1f}%)")

# Check distribution
print(f"\nTarget value distribution (percentiles):")
percentiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
for p in percentiles:
    print(f"  {p*100:3.0f}th percentile: {target.quantile(p):.0f}")

# =============================================
# 3. FEATURE ANALYSIS
# =============================================
print("\n\n🔍 3. FEATURE ANALYSIS")
print("-" * 40)

# Check numeric features
numeric_features = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 
                    'Visibility (10m)', 'Dew point temperature(°C)', 
                    'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']

print("Numeric features summary:")
for feature in numeric_features:
    if feature in df.columns:
        print(f"\n{feature}:")
        print(f"  Range: [{df[feature].min():.2f}, {df[feature].max():.2f}]")
        print(f"  Mean: {df[feature].mean():.2f}, Std: {df[feature].std():.2f}")
        
        # Check for unusual values
        if df[feature].min() < -50 or df[feature].max() > 100:
            print(f"  ⚠️  SUSPICIOUS: Extreme values!")
        if df[feature].isnull().sum() > 0:
            print(f"  ⚠️  WARNING: {df[feature].isnull().sum()} missing values")

# Check categorical features
print("\n\nCategorical features:")
categorical_features = ['Seasons', 'Holiday', 'Functioning Day']
for feature in categorical_features:
    if feature in df.columns:
        print(f"\n{feature}:")
        print(f"  Unique values: {df[feature].unique()}")
        value_counts = df[feature].value_counts()
        for val, count in value_counts.items():
            print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")

# =============================================
# 4. CHECK ORIGINAL MODEL'S TRAINING
# =============================================
print("\n\n🤖 4. REPRODUCE ORIGINAL MODEL (RMSE=112.5)")
print("-" * 40)

try:
    # Prepare data as original model might have done
    X = df.drop(['Rented Bike Count', 'Date'], axis=1, errors='ignore')
    
    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=['Seasons', 'Holiday', 'Functioning Day'], drop_first=True)
    
    y = df['Rented Bike Count'].values
    
    print(f"Features after encoding: {X.shape[1]}")
    print(f"Feature names: {list(X.columns)}")
    
    # Simple train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Train simple model (similar to original)
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42, 
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"\nReproduced model performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  Original claimed RMSE: 112.5")
    print(f"  Difference: {abs(rmse - 112.5):.2f}")
    
    # Calculate baseline (predicting mean)
    mean_prediction = np.full_like(y_test, y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_test, mean_prediction))
    print(f"\nBaseline (always predict mean={y_train.mean():.1f}):")
    print(f"  RMSE: {baseline_rmse:.2f}")
    
    if rmse < baseline_rmse:
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        print(f"  ✅ Model is {improvement:.1f}% better than baseline")
    else:
        print(f"  🚨 MODEL IS WORSE THAN BASELINE!")
    
    if abs(rmse - 112.5) > 20:
        print("\n⚠️  WARNING: Cannot reproduce original RMSE of 112.5!")
        print(f"   Our RMSE: {rmse:.2f}, Difference: {abs(rmse - 112.5):.2f}")
    else:
        print(f"\n✅ Successfully reproduced original RMSE (~{rmse:.2f})")
    
except Exception as e:
    print(f"Error reproducing model: {e}")
    import traceback
    traceback.print_exc()

# =============================================
# 5. CHECK FOR DATA LEAKAGE IN ORIGINAL DATA
# =============================================
print("\n\n🔎 5. CHECK FOR DATA ISSUES")
print("-" * 40)

# Check date column
if 'Date' in df.columns:
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Missing dates: {df['Date'].isnull().sum()}")
        
        # Check for duplicates
        date_hour_duplicates = df.duplicated(subset=['Date', 'Hour']).sum()
        if date_hour_duplicates > 0:
            print(f"⚠️  Duplicate date-hour combinations: {date_hour_duplicates}")
        else:
            print("✅ No duplicate date-hour combinations")
            
    except Exception as e:
        print(f"Error parsing dates: {e}")

# Check correlation with hour
if 'Hour' in df.columns:
    correlation = df['Rented Bike Count'].corr(df['Hour'])
    print(f"\nCorrelation with Hour: {correlation:.3f}")
    
    # Show average bikes by hour
    print("\nAverage bikes by hour:")
    hourly_avg = df.groupby('Hour')['Rented Bike Count'].mean()
    for hour in [6, 7, 8, 9, 12, 17, 18, 19, 22]:  # Key hours
        if hour in hourly_avg.index:
            print(f"  Hour {hour:2d}: {hourly_avg[hour]:6.1f} bikes")

# Check correlation with temperature
if 'Temperature(°C)' in df.columns:
    correlation = df['Rented Bike Count'].corr(df['Temperature(°C)'])
    print(f"\nCorrelation with Temperature: {correlation:.3f}")

# =============================================
# 6. SIMPLE MODEL TEST
# =============================================
print("\n\n🎯 6. SIMPLE MODEL TEST (Hour + Temperature only)")
print("-" * 40)

try:
    # Just use Hour and Temperature
    simple_features = ['Hour', 'Temperature(°C)']
    X_simple = df[simple_features]
    y_simple = df['Rented Bike Count'].values
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_simple, y_simple, test_size=0.2, random_state=42, shuffle=True
    )
    
    simple_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    simple_model.fit(X_train_s, y_train_s)
    
    y_pred_s = simple_model.predict(X_test_s)
    rmse_simple = np.sqrt(mean_squared_error(y_test_s, y_pred_s))
    
    print(f"Using only {simple_features}:")
    print(f"  RMSE: {rmse_simple:.2f}")
    
    if 'rmse' in locals():
        if rmse_simple < rmse:
            print(f"  ⚠️  SIMPLER model is BETTER! (by {rmse - rmse_simple:.2f} RMSE)")
        else:
            print(f"  ✅ Full model is better by {rmse_simple - rmse:.2f} RMSE")
    
except Exception as e:
    print(f"Error in simple model test: {e}")

# =============================================
# 7. CRITICAL FINDINGS
# =============================================
print("\n\n🚨🚨🚨 7. CRITICAL FINDINGS 🚨🚨🚨")
print("-" * 40)

print("SUMMARY:")
print(f"1. Target: Mean={target.mean():.1f}, Range=[{target.min()}, {target.max()}]")
print(f"2. Original model RMSE: 112.5 (claimed)")
print(f"3. Our reproduction RMSE: {rmse if 'rmse' in locals() else 'N/A':.2f}")

if 'rmse' in locals() and 'baseline_rmse' in locals():
    if rmse > baseline_rmse:
        print("\n🚨🚨🚨 CATASTROPHIC FINDING:")
        print("   Our model is WORSE than just predicting the average every time!")
        print("   This means our features are HURTING the predictions!")
    elif rmse > 200:
        print("\n⚠️  SERIOUS PROBLEM:")
        print(f"   RMSE={rmse:.2f} is too high (>200)")
        print("   Original model claims 112.5, we're much worse")
    elif rmse > 150:
        print("\n⚠️  PROBLEM:")
        print(f"   RMSE={rmse:.2f} is worse than original (112.5)")
    elif rmse > 120:
        print(f"\n📊 OK: RMSE={rmse:.2f} (close to original 112.5)")
    else:
        print(f"\n✅ GOOD: RMSE={rmse:.2f} (better than original 112.5!)")

# =============================================
# 8. EMERGENCY RECOMMENDATIONS
# =============================================
print("\n\n🎯 8. EMERGENCY RECOMMENDATIONS")
print("-" * 40)

print("IMMEDIATE ACTIONS:")
print("1. 🔄 Use the EXACT same preprocessing as original model")
print("2. 📊 Verify train/test split method (shuffle=True/False?)")
print("3. 🎯 Set target: First match 112.5, THEN aim for ≤30")
print("4. 🧪 Test with ONLY Hour + Temperature first")
print("5. 🔧 Check RandomForest parameters match original")

print("\nSUSPECTED ISSUES:")
print("1. Different train/test split (time-based vs random)")
print("2. Different feature encoding")
print("3. Different RandomForest hyperparameters")
print("4. Data shuffling differences")
print("\n" + "=" * 80)
print("🚨 DIAGNOSTICS COMPLETE")
print("=" * 80)

# Quick test prediction
print("\n🔮 Quick prediction test:")
print("Hour=14, Temperature=22.5°C, Summer, No Holiday, Functioning Day=Yes")
print("Expected from web app: ~1344 bikes")