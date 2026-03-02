"""
train_model_with_features.py
Feature-engineered Random Forest model for bike demand prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """Load and engineer features from raw data"""
    print("📂 Loading dataset...")
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    print(f"   Original shape: {df.shape}")
    
    # ========== STEP 1: DATE/TIME FEATURES ==========
    print("📅 Creating date/time features...")
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day_of_month'] = df['Date'].dt.day
    df['year'] = df['Date'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # ========== STEP 2: HOUR-BASED FEATURES ==========
    print("⏰ Creating hour-based features...")
    df['hour_squared'] = df['Hour'] ** 2
    df['hour_cubed'] = df['Hour'] ** 3
    
    def categorize_hour(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    df['hour_category'] = df['Hour'].apply(categorize_hour)
    df['is_peak_hour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)) | ((df['Hour'] >= 17) & (df['Hour'] <= 19))
    df['is_peak_hour'] = df['is_peak_hour'].astype(int)
    
    def rush_hour_intensity(hour):
        if hour in [7, 8, 9, 17, 18, 19]:
            return 1.0
        elif hour in [6, 10, 16, 20]:
            return 0.5
        else:
            return 0.0
    
    df['rush_hour_intensity'] = df['Hour'].apply(rush_hour_intensity)
    
    # ========== STEP 3: INTERACTION TERMS ==========
    print("🔄 Creating interaction features...")
    df['temp_hour_interaction'] = df['Temperature(°C)'] * df['Hour']
    df['temp_humidity_interaction'] = df['Temperature(°C)'] * df['Humidity(%)']
    
    # Wind chill formula
    df['wind_chill'] = 13.12 + 0.6215*df['Temperature(°C)'] - 11.37*(df['Wind speed (m/s)']**0.16) + 0.3965*df['Temperature(°C)']*(df['Wind speed (m/s)']**0.16)
    
    # Heat index approximation
    def calculate_heat_index(temp, humidity):
        if temp < 27:
            return temp
        else:
            HI = -42.379 + 2.04901523*temp + 10.14333127*humidity - 0.22475541*temp*humidity - 0.00683783*(temp**2) - 0.05481717*(humidity**2) + 0.00122874*(temp**2)*humidity + 0.00085282*temp*(humidity**2) - 0.00000199*(temp**2)*(humidity**2)
            return HI
    
    df['heat_index'] = df.apply(lambda row: calculate_heat_index(row['Temperature(°C)'], row['Humidity(%)']), axis=1)
    
    # ========== STEP 4: WEATHER CATEGORIZATION ==========
    print("🌤️ Creating weather features...")
    def calculate_weather_severity(row):
        score = 0
        
        # Temperature effect
        temp = row['Temperature(°C)']
        if 20 <= temp <= 25:
            score += 0
        elif 15 <= temp < 20 or 25 < temp <= 30:
            score += 1
        elif 10 <= temp < 15 or 30 < temp <= 35:
            score += 2
        else:
            score += 3
        
        # Rain effect
        if row['Rainfall(mm)'] > 0:
            score += min(3, row['Rainfall(mm)'] * 2)
        
        # Snow effect
        if row['Snowfall (cm)'] > 0:
            score += min(3, row['Snowfall (cm)'] * 1)
        
        # Wind effect
        if row['Wind speed (m/s)'] > 5:
            score += 2
        
        # Visibility effect
        if row['Visibility (10m)'] < 1000:
            score += 2
        
        return score
    
    df['weather_severity_score'] = df.apply(calculate_weather_severity, axis=1)
    
    def categorize_weather(score):
        if score == 0:
            return 'perfect'
        elif 1 <= score <= 2:
            return 'good'
        elif 3 <= score <= 5:
            return 'fair'
        elif 6 <= score <= 8:
            return 'poor'
        else:
            return 'severe'
    
    df['weather_category'] = df['weather_severity_score'].apply(categorize_weather)
    df['comfort_index'] = 10 - df['weather_severity_score']
    df['comfort_index'] = df['comfort_index'].clip(lower=0)
    
    # ========== STEP 5: LAG FEATURES ==========
    print("📈 Creating lag features...")
    df = df.sort_values(['Date', 'Hour'])
    df['demand_lag_24'] = df['Rented Bike Count'].shift(24)
    df['demand_lag_168'] = df['Rented Bike Count'].shift(168)
    df['demand_rolling_3h'] = df['Rented Bike Count'].rolling(window=3, min_periods=1).mean()
    df['demand_rolling_24h'] = df['Rented Bike Count'].rolling(window=24, min_periods=1).mean()
    
    # Fill NaN values
    df['demand_lag_24'] = df['demand_lag_24'].bfill().ffill()
    df['demand_lag_168'] = df['demand_lag_168'].bfill().ffill()
    df['demand_rolling_3h'] = df['demand_rolling_3h'].bfill().ffill()
    df['demand_rolling_24h'] = df['demand_rolling_24h'].bfill().ffill()
    
    # ========== STEP 6: SEASONAL FEATURES ==========
    print("🍂 Creating seasonal features...")
    season_avg_temp = df.groupby('Seasons')['Temperature(°C)'].mean().to_dict()
    df['season_avg_temp'] = df['Seasons'].map(season_avg_temp)
    df['temp_deviation_from_season'] = df['Temperature(°C)'] - df['season_avg_temp']
    
    def estimate_daylight_hour(hour, month):
        if month in [12, 1, 2]:
            return 1 if 8 <= hour <= 16 else 0
        elif month in [3, 4, 5]:
            return 1 if 7 <= hour <= 18 else 0
        elif month in [6, 7, 8]:
            return 1 if 6 <= hour <= 20 else 0
        else:
            return 1 if 7 <= hour <= 18 else 0
    
    df['is_daylight'] = df.apply(lambda row: estimate_daylight_hour(row['Hour'], row['month']), axis=1)
    df['workday_hour_interaction'] = df['Hour'] * (1 - df['is_weekend'])
    
    # ========== STEP 7: ONE-HOT ENCODING ==========
    print("🔤 Encoding categorical features...")
    categorical_cols = ['hour_category', 'weather_category', 'Seasons']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # ========== STEP 8: PREPARE FINAL FEATURES ==========
    print("📊 Preparing final feature set...")
    feature_columns = [
        'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 
        'Visibility (10m)', 'Dew point temperature(°C)', 
        'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)',
        'hour_squared', 'hour_cubed', 'is_peak_hour', 'rush_hour_intensity',
        'is_weekend', 'day_of_week', 'temp_hour_interaction', 'temp_humidity_interaction',
        'wind_chill', 'heat_index', 'weather_severity_score', 'comfort_index',
        'demand_lag_24', 'demand_lag_168', 'demand_rolling_3h', 'demand_rolling_24h',
        'temp_deviation_from_season', 'is_daylight', 'workday_hour_interaction'
    ]
    
    # Add one-hot encoded columns
    one_hot_cols = [col for col in df_encoded.columns if any(cat in col for cat in categorical_cols)]
    feature_columns.extend(one_hot_cols)
    
    # Filter to only columns that exist
    feature_columns = [col for col in feature_columns if col in df_encoded.columns]
    
    X = df_encoded[feature_columns]
    y = df_encoded['Rented Bike Count']
    
    print(f"✅ Final feature set: {len(feature_columns)} features")
    print(f"✅ Dataset shape: X={X.shape}, y={y.shape}")
    
    return X, y, feature_columns

def train_random_forest(X, y, feature_names):
    """Train Random Forest with hyperparameter tuning"""
    print("\n🎯 Training Random Forest model...")
    
    # Use time-series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Train-test split (preserve time order)
    test_size = int(len(X) * 0.2)
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Initialize model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, verbose=0)
    
    # Hyperparameter grid (reduced for faster training)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    
    # Grid search with time-series cross-validation
    print("   Performing grid search...")
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=tscv, 
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_rf = grid_search.best_estimator_
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Predictions
    y_pred = best_rf.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   R² Score: {r2:.3f}")
    print(f"   MAPE: {mape:.1f}%")
    
    print(f"\n⚙️ BEST PARAMETERS:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"\n🏆 TOP 10 FEATURE IMPORTANCES:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    return best_rf, rmse, mae, r2, mape, feature_importance

def save_model(model, feature_names, metrics, feature_importance, filepath='models/improved_random_forest.pkl'):
    """Save model and metadata"""
    print(f"\n💾 Saving model to {filepath}...")
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'training_date': pd.Timestamp.now().isoformat(),
        'metrics': metrics,
        'feature_importance': feature_importance.to_dict()
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to {filepath}")
    
    # Save metrics to JSON
    metrics_dict = {
        'rmse': float(metrics[0]),
        'mae': float(metrics[1]),
        'r2': float(metrics[2]),
        'mape': float(metrics[3]),
        'n_features': len(feature_names),
        'model_type': 'RandomForestRegressor',
        'best_params': str(model.get_params()),
        'feature_importance': feature_importance.head(20).to_dict()
    }
    
    with open('models/improved_model_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print("✅ Metrics saved to models/improved_model_metrics.json")
    
    return model_data

def compare_with_baseline(new_rmse):
    """Compare with the original model's RMSE (112.5)"""
    baseline_rmse = 112.5
    improvement = ((baseline_rmse - new_rmse) / baseline_rmse) * 100
    
    print(f"\n📈 COMPARISON WITH BASELINE:")
    print(f"   Baseline RMSE: {baseline_rmse}")
    print(f"   New RMSE: {new_rmse:.2f}")
    print(f"   Improvement: {improvement:.1f}%")
    
    if new_rmse <= 30:
        print(f"🎉🎉🎉 SUCCESS! RMSE = {new_rmse:.2f} ≤ 30! 🎉🎉🎉")
        return True
    else:
        print(f"⚠️  RMSE = {new_rmse:.2f} > 30. More work needed.")
        return False

if __name__ == "__main__":
    print("="*60)
    print("🚀 BIKE DEMAND PREDICTION - FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data('data/raw/seoul_bike.csv')
    
    # Train model
    model, rmse, mae, r2, mape, feature_importance = train_random_forest(X, y, feature_names)
    
    # Compare with baseline
    success = compare_with_baseline(rmse)
    
    # Save model
    model_data = save_model(model, feature_names, (rmse, mae, r2, mape), feature_importance)
    
    # Print summary
    print("\n" + "="*60)
    print("📋 PROJECT SUMMARY")
    print("="*60)
    print(f"Model: Random Forest Regressor")
    print(f"Features: {len(feature_names)} engineered features")
    print(f"Performance: RMSE={rmse:.2f}, R²={r2:.3f}")
    print(f"Target RMSE ≤ 30: {'✅ ACHIEVED' if success else '❌ NOT ACHIEVED'}")
    print(f"Model saved: models/improved_random_forest.pkl")
    
    if not success:
        print("\n" + "="*60)
        print("🔄 NEXT STEPS TO IMPROVE RMSE:")
        print("="*60)
        print("1. Try XGBoost model (usually better than Random Forest)")
        print("2. Add more domain-specific features")
        print("3. Try deep learning (LSTM) for time-series patterns")
        print("4. Ensemble multiple models")
        print("5. More extensive hyperparameter tuning")