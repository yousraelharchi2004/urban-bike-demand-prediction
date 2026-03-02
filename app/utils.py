"""
app/utils.py - Complete updated version with advanced feature engineering
FIXED: Better field name handling for Korean characters
"""

import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
import json
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_type='original'):
    """Load either original, improved, advanced, or xgboost model"""
    import joblib
    import os
    
    if model_type == 'xgboost':
        model_paths = [
            "models/xgboost_model.pkl",
            "../models/xgboost_model.pkl",
            os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_model.pkl"),
            os.path.join(os.getcwd(), "models", "xgboost_model.pkl")
        ]
        model_name = "XGBoost"
    elif model_type == 'improved':
        model_paths = [
            "models/improved_random_forest.pkl",
            "../models/improved_random_forest.pkl",
            os.path.join(os.path.dirname(__file__), "..", "models", "improved_random_forest.pkl"),
            os.path.join(os.getcwd(), "models", "improved_random_forest.pkl")
        ]
        model_name = "Improved Random Forest"
    elif model_type == 'advanced':
        model_paths = [
            "models/advanced_random_forest.pkl",
            "../models/advanced_random_forest.pkl",
            os.path.join(os.path.dirname(__file__), "..", "models", "advanced_random_forest.pkl"),
            os.path.join(os.getcwd(), "models", "advanced_random_forest.pkl")
        ]
        model_name = "Advanced Random Forest"
    else:
        model_paths = [
            "models/random_forest_model.pkl",
            "../models/random_forest_model.pkl",
            os.path.join(os.path.dirname(__file__), "..", "models", "random_forest_model.pkl"),
            os.path.join(os.getcwd(), "models", "random_forest_model.pkl")
        ]
        model_name = "Random Forest"
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ Found {model_name} model at: {model_path}")
            try:
                model_data = joblib.load(model_path)
                
                # HANDLE IMPROVED MODEL SPECIFICALLY - IT'S STORED AS A DICTIONARY
                if model_type == 'improved' and isinstance(model_data, dict):
                    print(f"📦 Improved model loaded as dictionary. Extracting actual model...")
                    if 'model' in model_data:
                        model = model_data['model']
                        print(f"✅ Extracted Random Forest model from dictionary key 'model'")
                        print(f"   Model type: {type(model).__name__}")
                        
                        if hasattr(model, 'feature_names_in_'):
                            feature_names = list(model.feature_names_in_)
                            print(f"   Model expects {len(feature_names)} features")
                            print(f"   Feature names: {feature_names}")
                        elif hasattr(model, 'n_features_in_'):
                            print(f"   Model expects {model.n_features_in_} features")
                        return model
                    else:
                        print(f"❌ No 'model' key found in dictionary")
                        print(f"   Available keys: {list(model_data.keys())}")
                        # Fallback: try to find any object with predict method
                        for key, value in model_data.items():
                            if hasattr(value, 'predict'):
                                print(f"✅ Found model in key '{key}', type: {type(value).__name__}")
                                return value
                        print(f"❌ No valid model object found in dictionary")
                        return None
                else:
                    # For original, advanced, and xgboost models
                    model = model_data
                    print(f"✅ {model_name} loaded successfully. Type: {type(model).__name__}")
                    
                    if hasattr(model, 'feature_names_in_'):
                        feature_names = list(model.feature_names_in_)
                        print(f"✅ Model feature names: {feature_names}")
                        print(f"✅ Model expects {len(feature_names)} features")
                    elif hasattr(model, 'n_features_in_'):
                        print(f"✅ Model expects {model.n_features_in_} features")
                    return model
                    
            except Exception as e:
                print(f"❌ Error loading model from {model_path}: {e}")
                traceback.print_exc()
                continue
    
    # If we get here, no model was found
    if model_type == 'xgboost':
        raise FileNotFoundError(f"❌ {model_name} model not found. Please run the XGBoost training script first.")
    elif model_type == 'improved':
        raise FileNotFoundError(f"❌ {model_name} model not found. Please run the training script first.")
    elif model_type == 'advanced':
        raise FileNotFoundError(f"❌ {model_name} model not found. Please run train_with_advanced_features.py first.")
    else:
        raise FileNotFoundError("❌ Model file not found. Please train the model first.")

def load_advanced_model():
    """Load the advanced Random Forest model"""
    return load_model('advanced')

def preprocess_input(data, use_engineered_features=False, model_type='original', model=None):
    """
    Convert input JSON to DataFrame with the EXACT same columns used in training.
    
    Args:
        data: Input dictionary with feature values
        use_engineered_features: Whether to use feature engineering
        model_type: 'original', 'improved', 'advanced', or 'xgboost'
        model: The loaded model object (optional, used to get feature_names_in_)
    """
    print(f"🔄 Preprocessing input for {model_type.upper()} model...")
    
    # Create a copy to avoid modifying original
    df = pd.DataFrame([data])
    
    # Debug: print what columns we received
    print(f"📥 Input data columns received: {list(df.columns)}")
    print(f"📥 Input data shape: {df.shape}")
    
    if model_type == 'advanced' or model_type == 'xgboost':
        return preprocess_with_advanced_features(df, model)
    elif model_type == 'improved':
        return preprocess_with_engineered_features(df, model)
    else:
        return preprocess_original_features(df, model)

def preprocess_original_features(input_data, model=None):
    """Preprocess for original 14-feature model - handles both dict and DataFrame"""
    # Handle if input is a dictionary
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data
    
    # Get expected features from model if available, otherwise use default
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        print(f"📋 Using model.feature_names_in_ for original model: {expected_features}")
    else:
        # FALLBACK: Use the model's actual training feature names
        # The original model was trained with Temperature(캜) and Dew point temperature(캜)
        expected_features = [
            'Hour',
            'Temperature(캜)',  # ORIGINAL: Model was trained with (캜) not (°C)
            'Humidity(%)',
            'Wind speed (m/s)',
            'Visibility (10m)',
            'Dew point temperature(캜)',  # ORIGINAL: Model was trained with (캜) not (°C)
            'Solar Radiation (MJ/m2)',
            'Rainfall(mm)',
            'Snowfall (cm)',
            'Seasons_Spring',
            'Seasons_Summer',
            'Seasons_Winter',
            'Holiday_No Holiday',
            'Functioning Day_Yes'
        ]
        print(f"📋 Using default feature list for original model: {expected_features}")
    
    print(f"📋 Original model expects these {len(expected_features)} features: {expected_features}")
    
    # Handle field name mappings - convert input format to model's expected format
    # If input has Temperature(°C), rename it to Temperature(캜) for the model
    if 'Temperature(°C)' in df.columns and 'Temperature(캜)' in expected_features:
        df['Temperature(캜)'] = df['Temperature(°C)']
    
    # If input has Dew point temperature(°C), rename it to Dew point temperature(캜)
    if 'Dew point temperature(°C)' in df.columns and 'Dew point temperature(캜)' in expected_features:
        df['Dew point temperature(캜)'] = df['Dew point temperature(°C)']
    
    # Also handle reverse mapping (if input has old format but model expects new format)
    if 'Temperature(캜)' in df.columns and 'Temperature(°C)' in expected_features:
        df['Temperature(°C)'] = df['Temperature(캜)']
    
    if 'Dew point temperature(캜)' in df.columns and 'Dew point temperature(°C)' in expected_features:
        df['Dew point temperature(°C)'] = df['Dew point temperature(캜)']
    
    # Handle Seasons string input
    if 'Seasons' in df.columns:
        season_value = str(df['Seasons'].iloc[0]).lower() if not df['Seasons'].empty else ''
        df['Seasons_Spring'] = 1 if 'spring' in season_value else 0
        df['Seasons_Summer'] = 1 if 'summer' in season_value else 0
        df['Seasons_Winter'] = 1 if 'winter' in season_value else 0
    
    # Handle Holiday string input
    if 'Holiday' in df.columns:
        holiday_value = str(df['Holiday'].iloc[0]).lower() if not df['Holiday'].empty else ''
        # "No Holiday" means it's not a holiday, so Holiday_No Holiday = 1
        df['Holiday_No Holiday'] = 1 if 'holiday' not in holiday_value or 'no' in holiday_value else 0
    
    # Handle Functioning Day string input
    if 'Functioning Day' in df.columns:
        functioning_value = str(df['Functioning Day'].iloc[0]).lower() if not df['Functioning Day'].empty else ''
        df['Functioning Day_Yes'] = 1 if 'yes' in functioning_value else 0
    
    # Check for missing columns
    missing_cols = [col for col in expected_features if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_features]
    
    if missing_cols:
        print(f"⚠️ Missing columns in input: {missing_cols}")
    if extra_cols:
        print(f"⚠️ Extra columns in input (will be ignored): {extra_cols}")
    
    # Create a new DataFrame with exactly the expected features
    processed_df = pd.DataFrame()
    
    for feature in expected_features:
        if feature in df.columns:
            processed_df[feature] = df[feature]
        else:
            # Add missing column with default value 0
            processed_df[feature] = 0
            print(f"➕ Added missing column '{feature}' with value 0")
    
    # Ensure correct data types
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Fill any NaN values with 0
    processed_df = processed_df.fillna(0)
    
    # Ensure columns are in the exact order expected by the model
    if model is not None and hasattr(model, 'feature_names_in_'):
        processed_df = processed_df[list(model.feature_names_in_)]
    
    print(f"✅ Processed data shape: {processed_df.shape}")
    print(f"✅ Processed columns: {list(processed_df.columns)}")
    print(f"✅ First row values: {processed_df.iloc[0].to_dict()}")
    
    return processed_df

def preprocess_input_original(input_data):
    """Alias for preprocess_original_features for compatibility"""
    return preprocess_original_features(input_data)

def prepare_input_with_engineered_features(input_data):
    """
    Prepare input data with all engineered features for improved model
    """
    # Convert to DataFrame if dict
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data
    
    # FIX: Handle both temperature field formats
    if 'Temperature(캜)' in df.columns and 'Temperature(°C)' not in df.columns:
        df['Temperature(°C)'] = df['Temperature(캜)']
    
    # FIX: Handle both dew point field formats
    if 'Dew point temperature(캜)' in df.columns and 'Dew point temperature(°C)' not in df.columns:
        df['Dew point temperature(°C)'] = df['Dew point temperature(캜)']
    
    # Convert day_of_week to integer (fix for string input from form)
    if 'day_of_week' in df.columns:
        try:
            # Handle string conversion safely
            if isinstance(df['day_of_week'].iloc[0], str):
                df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype(int)
            else:
                df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype(int)
        except Exception as e:
            print(f"⚠️ Error converting day_of_week: {e}")
            df['day_of_week'] = 0
    
    # Basic engineered features
    df['hour_squared'] = df['Hour'] ** 2
    df['hour_cubed'] = df['Hour'] ** 3
    
    # Weekend flag - FIXED: ensure day_of_week is integer before comparison
    try:
        day_of_week_val = int(df.get('day_of_week', 0).iloc[0])
    except (ValueError, TypeError):
        day_of_week_val = 0
    
    # FIXED: Convert boolean to integer properly
    df['is_weekend'] = 1 if day_of_week_val >= 5 else 0
    
    # Peak hours
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
    
    # Interaction terms - USE Temperature(°C) which we now have
    df['temp_hour_interaction'] = df['Temperature(°C)'] * df['Hour']
    df['temp_humidity_interaction'] = df['Temperature(°C)'] * df['Humidity(%)']
    
    # Wind chill
    df['wind_chill'] = 13.12 + 0.6215*df['Temperature(°C)'] - 11.37*(df['Wind speed (m/s)']**0.16) + 0.3965*df['Temperature(°C)']*(df['Wind speed (m/s)']**0.16)
    
    # Weather severity
    def calculate_weather_severity(row):
        score = 0
        temp = row['Temperature(°C)']
        
        if 20 <= temp <= 25:
            score += 0
        elif 15 <= temp < 20 or 25 < temp <= 30:
            score += 1
        elif 10 <= temp < 15 or 30 < temp <= 35:
            score += 2
        else:
            score += 3
        
        if row.get('Rainfall(mm)', 0) > 0:
            score += min(3, row['Rainfall(mm)'] * 2)
        
        if row.get('Snowfall (cm)', 0) > 0:
            score += min(3, row['Snowfall (cm)'] * 1)
        
        if row['Wind speed (m/s)'] > 5:
            score += 2
        
        if row['Visibility (10m)'] < 1000:
            score += 2
        
        return score
    
    df['weather_severity_score'] = df.apply(calculate_weather_severity, axis=1)
    df['comfort_index'] = 10 - df['weather_severity_score']
    df['comfort_index'] = df['comfort_index'].clip(lower=0)
    
    # Handle categorical features
    # Handle Seasons string input
    if 'Seasons' in df.columns:
        season_value = str(df['Seasons'].iloc[0]).lower() if not df['Seasons'].empty else ''
        df['Seasons_Spring'] = 1 if 'spring' in season_value else 0
        df['Seasons_Summer'] = 1 if 'summer' in season_value else 0
        df['Seasons_Winter'] = 1 if 'winter' in season_value else 0
    
    # Handle Holiday string input
    if 'Holiday' in df.columns:
        holiday_value = str(df['Holiday'].iloc[0]).lower() if not df['Holiday'].empty else ''
        df['Holiday_No Holiday'] = 1 if 'holiday' not in holiday_value or 'no' in holiday_value else 0
    
    # Handle Functioning Day string input
    if 'Functioning Day' in df.columns:
        functioning_value = str(df['Functioning Day'].iloc[0]).lower() if not df['Functioning Day'].empty else ''
        df['Functioning Day_Yes'] = 1 if 'yes' in functioning_value else 0
    
    # Add hour category features
    df['hour_category_evening'] = ((df['Hour'] >= 17) & (df['Hour'] <= 21)).astype(int)
    df['hour_category_morning'] = ((df['Hour'] >= 6) & (df['Hour'] <= 9)).astype(int)
    df['hour_category_night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)
    
    # Add weather category features
    df['weather_category_good'] = ((df['Temperature(°C)'] >= 15) & (df['Temperature(°C)'] <= 25) & (df['Rainfall(mm)'] == 0)).astype(int)
    df['weather_category_perfect'] = ((df['Temperature(°C)'] >= 18) & (df['Temperature(°C)'] <= 22) & (df['Rainfall(mm)'] == 0) & (df['Wind speed (m/s)'] < 3)).astype(int)
    df['weather_category_poor'] = ((df['Rainfall(mm)'] > 0) | (df['Snowfall (cm)'] > 0)).astype(int)
    df['weather_category_severe'] = ((df['Wind speed (m/s)'] > 10) | (df['Temperature(°C)'] < 0) | (df['Temperature(°C)'] > 35)).astype(int)
    
    return df

def preprocess_with_engineered_features(input_data, model=None):
    """Preprocess with all engineered features (38 features) for improved model"""
    # First apply feature engineering
    engineered_df = prepare_input_with_engineered_features(input_data)
    
    # Get expected features from model if available, otherwise use default list
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        print(f"📋 Using model.feature_names_in_ for improved model: {len(expected_features)} features")
        print(f"📋 Feature order from model: {expected_features}")
    else:
        expected_features = get_engineered_feature_list()
        print(f"📋 Using default feature list for improved model: {len(expected_features)} features")
    
    print(f"📋 Improved model expects {len(expected_features)} features")
    
    # Debug: Check which features are missing
    missing_features = [f for f in expected_features if f not in engineered_df.columns]
    if missing_features:
        print(f"⚠️ Missing features that will be added with defaults: {missing_features}")
    
    # Create a new DataFrame with exactly the expected features in the EXACT order
    processed_df = pd.DataFrame(index=engineered_df.index)
    
    for feature in expected_features:
        if feature in engineered_df.columns:
            processed_df[feature] = engineered_df[feature]
        else:
            # Add missing column with appropriate default value
            if feature in ['demand_lag_24', 'demand_lag_168', 'demand_rolling_3h', 'demand_rolling_24h']:
                processed_df[feature] = 500  # Default demand value
            elif feature in ['heat_index', 'wind_chill']:
                # Use temperature from engineered_df if available
                if 'Temperature(°C)' in engineered_df.columns:
                    processed_df[feature] = engineered_df['Temperature(°C)']
                else:
                    processed_df[feature] = 20
            elif feature in ['temp_deviation_from_season']:
                processed_df[feature] = 0  # Default deviation
            else:
                processed_df[feature] = 0
            print(f"➕ Added missing column '{feature}' with default value")
    
    # Ensure correct data types
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Fill any NaN values with 0
    processed_df = processed_df.fillna(0)
    
    # The DataFrame is already in the correct order because we built it sequentially
    print(f"✅ Processed data shape: {processed_df.shape}")
    print(f"✅ Processed columns (in exact order): {list(processed_df.columns)}")
    print(f"✅ Total engineered features: {len(processed_df.columns)}")
    
    return processed_df

def preprocess_input_improved(input_data):
    """Alias for preprocess_with_engineered_features for compatibility"""
    # First, handle field name mapping
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data
    
    # Map Korean temperature to degree symbol if needed
    if 'Temperature(캜)' in df.columns and 'Temperature(°C)' not in df.columns:
        df['Temperature(°C)'] = df['Temperature(캜)']
    
    # Map Korean dew point to degree symbol if needed
    if 'Dew point temperature(캜)' in df.columns and 'Dew point temperature(°C)' not in df.columns:
        df['Dew point temperature(°C)'] = df['Dew point temperature(캜)']
    
    return preprocess_with_engineered_features(df)

def prepare_input_with_advanced_features(input_data):
    """
    Prepare input data with ADVANCED feature engineering for RMSE reduction
    FIXED: Creating exact feature names that match the model
    """
    # Convert to DataFrame if dict
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data
    
    print(f"🔧 Preparing advanced features from columns: {list(df.columns)}")
    
    # ==================================================
    # MAP ALL POSSIBLE FIELD NAMES TO STANDARD FORMAT
    # ==================================================
    
    # Map Hour
    if 'Hour' in df.columns and 'hour' not in df.columns:
        df['hour'] = df['Hour']
    elif 'hour' in df.columns and 'Hour' not in df.columns:
        df['Hour'] = df['hour']
    
    # Map Temperature
    if 'Temperature(캜)' in df.columns:
        df['Temperature(°C)'] = df['Temperature(캜)']
        df['temperature'] = df['Temperature(캜)']
    elif 'Temperature(°C)' in df.columns:
        df['temperature'] = df['Temperature(°C)']
    elif 'temperature' in df.columns:
        df['Temperature(°C)'] = df['temperature']
    
    # Map Dew Point
    if 'Dew point temperature(캜)' in df.columns:
        df['Dew point temperature(°C)'] = df['Dew point temperature(캜)']
        df['dew_point'] = df['Dew point temperature(캜)']
    elif 'Dew point temperature(°C)' in df.columns:
        df['dew_point'] = df['Dew point temperature(°C)']
    elif 'dew_point' in df.columns:
        df['Dew point temperature(°C)'] = df['dew_point']
    
    # Map Humidity - CRITICAL FIX: Create EXACT name 'Humidity%' not 'Humidity(%)'
    if 'Humidity(%)' in df.columns:
        df['Humidity%'] = df['Humidity(%)']  # Remove parentheses
        df['humidity'] = df['Humidity(%)']
    elif 'Humidity%' in df.columns:
        df['Humidity(%)'] = df['Humidity%']
        df['humidity'] = df['Humidity%']
    elif 'humidity' in df.columns:
        df['Humidity(%)'] = df['humidity']
        df['Humidity%'] = df['humidity']  # Add both formats
    
    # Map Wind Speed - CRITICAL FIX: Create EXACT name 'Wind_speed_m/s'
    if 'Wind speed (m/s)' in df.columns:
        df['Wind_speed_m/s'] = df['Wind speed (m/s)']
        df['wind_speed'] = df['Wind speed (m/s)']
    elif 'Wind_speed_m/s' in df.columns:
        df['Wind speed (m/s)'] = df['Wind_speed_m/s']
        df['wind_speed'] = df['Wind_speed_m/s']
    elif 'wind_speed' in df.columns:
        df['Wind speed (m/s)'] = df['wind_speed']
        df['Wind_speed_m/s'] = df['wind_speed']
    
    # Map Visibility
    if 'Visibility (10m)' in df.columns:
        df['visibility'] = df['Visibility (10m)']
    elif 'visibility' in df.columns:
        df['Visibility (10m)'] = df['visibility']
    
    # Map Solar Radiation - CRITICAL FIX: Create EXACT name 'Solar_Radiation_MJ/m2'
    if 'Solar Radiation (MJ/m2)' in df.columns:
        df['Solar_Radiation_MJ/m2'] = df['Solar Radiation (MJ/m2)']
        df['solar_radiation'] = df['Solar Radiation (MJ/m2)']
    elif 'Solar_Radiation_MJ/m2' in df.columns:
        df['Solar Radiation (MJ/m2)'] = df['Solar_Radiation_MJ/m2']
        df['solar_radiation'] = df['Solar_Radiation_MJ/m2']
    elif 'solar_radiation' in df.columns:
        df['Solar Radiation (MJ/m2)'] = df['solar_radiation']
        df['Solar_Radiation_MJ/m2'] = df['solar_radiation']
    
    # Map Rainfall
    if 'Rainfall(mm)' in df.columns:
        df['rainfall'] = df['Rainfall(mm)']
    elif 'rainfall' in df.columns:
        df['Rainfall(mm)'] = df['rainfall']
    
    # Map Snowfall
    if 'Snowfall (cm)' in df.columns:
        df['snowfall'] = df['Snowfall (cm)']
    elif 'snowfall' in df.columns:
        df['Snowfall (cm)'] = df['snowfall']
    
    # Map day_of_week
    if 'day_of_week' in df.columns:
        try:
            if isinstance(df['day_of_week'].iloc[0], str):
                df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype(int)
            else:
                df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype(int)
        except Exception as e:
            print(f"⚠️ Error converting day_of_week: {e}")
            df['day_of_week'] = 0
    
    # Ensure required numeric fields exist with EXACT names
    required_fields = {
        'hour': 12,
        'temperature': 20,
        'Humidity%': 50,  # Note: EXACT name with % not parentheses
        'Wind_speed_m/s': 2.5,  # Note: EXACT name
        'visibility': 2000,
        'rainfall': 0,
        'snowfall': 0,
        'Solar_Radiation_MJ/m2': 1.5,  # Note: EXACT name
        'dew_point': 10,
        'day_of_week': 3
    }
    
    for field, default in required_fields.items():
        if field not in df.columns:
            df[field] = default
            print(f"➕ Added missing field '{field}' with default value {default}")
    
    # Convert all to numeric
    for field in required_fields.keys():
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(default)
    
    # ==================================================
    # 1. HOUR SQUARED FEATURE (hour²)
    # ==================================================
    df['hour_squared'] = df['hour'] ** 2
    
    # ==================================================
    # 2. TEMPERATURE-HOUR INTERACTION TERM
    # ==================================================
    df['temp_hour_interaction'] = df['temperature'] * df['hour']
    
    # ==================================================
    # 3. WEEKDAY/WEEKEND FLAG
    # ==================================================
    if 'day_of_week' in df.columns:
        try:
            day_val = int(df['day_of_week'].iloc[0])
        except (ValueError, TypeError):
            day_val = 0
        df['is_weekend'] = 1 if day_val >= 5 else 0
    else:
        df['is_weekend'] = 0
    
    # ==================================================
    # 4. PEAK HOURS INDICATOR (7-9 AM, 5-7 PM)
    # ==================================================
    df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
    df['is_peak_hour'] = df['is_peak_hour'].astype(int)
    
    # ==================================================
    # 5. RUSH HOUR INTENSITY
    # ==================================================
    def rush_hour_intensity_advanced(hour):
        if hour in [7, 8, 17, 18]:  # Peak of peaks
            return 1.0
        elif hour in [6, 9, 16, 19]:  # Shoulder peaks
            return 0.7
        elif hour in [10, 15, 20]:  # Near peaks
            return 0.3
        else:
            return 0.0
    
    df['rush_hour_intensity'] = df['hour'].apply(rush_hour_intensity_advanced)
    
    # ==================================================
    # 6. TEMPERATURE SQUARED
    # ==================================================
    df['temp_squared'] = df['temperature'] ** 2
    
    # ==================================================
    # 7. GOOD BIKING CONDITIONS
    # ==================================================
    df['good_biking_conditions'] = (
        (df['temperature'] >= 10) & 
        (df['temperature'] <= 30) & 
        (df['rainfall'] == 0) & 
        (df['snowfall'] == 0) & 
        (df['Wind_speed_m/s'] < 10)
    ).astype(int)
    
    # ==================================================
    # 8. CYCLICAL TIME FEATURES
    # ==================================================
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # ==================================================
    # 9. SEASON ENCODING
    # ==================================================
    # Default to Summer if not specified
    if 'season_Spring' not in df.columns:
        df['season_Spring'] = 0
    if 'season_Summer' not in df.columns:
        df['season_Summer'] = 1
    if 'season_Winter' not in df.columns:
        df['season_Winter'] = 0
    if 'season_Autumn' not in df.columns:
        df['season_Autumn'] = 0
    
    # ==================================================
    # 10. HOLIDAY ENCODING
    # ==================================================
    if 'holiday_Holiday' not in df.columns:
        df['holiday_Holiday'] = 0
    if 'holiday_No Holiday' not in df.columns:
        df['holiday_No Holiday'] = 1
    
    # ==================================================
    # 11. FUNCTIONING DAY ENCODING
    # ==================================================
    if 'functioning_day_Yes' not in df.columns:
        df['functioning_day_Yes'] = 1
    if 'functioning_day_No' not in df.columns:
        df['functioning_day_No'] = 0
    
    # ==================================================
    # 12. TEMPERATURE CATEGORIES
    # ==================================================
    temp = df['temperature'].iloc[0]
    temp_cats = {
        'temp_category_freezing': 1 if temp < 0 else 0,
        'temp_category_cold': 1 if 0 <= temp < 10 else 0,
        'temp_category_cool': 1 if 10 <= temp < 20 else 0,
        'temp_category_warm': 1 if 20 <= temp < 30 else 0,
        'temp_category_hot': 1 if temp >= 30 else 0
    }
    
    for cat, val in temp_cats.items():
        df[cat] = val
    
    # ==================================================
    # 13. TIME OF DAY CATEGORIES
    # ==================================================
    hour = df['hour'].iloc[0]
    time_cats = {
        'time_of_day_night': 1 if hour < 6 else 0,
        'time_of_day_morning': 1 if 6 <= hour < 12 else 0,
        'time_of_day_afternoon': 1 if 12 <= hour < 18 else 0,
        'time_of_day_evening': 1 if hour >= 18 else 0
    }
    
    for cat, val in time_cats.items():
        df[cat] = val
    
    print(f"✅ Advanced feature engineering complete. Final columns: {list(df.columns)}")
    return df

def preprocess_with_advanced_features(input_data, model=None):
    """Preprocess with ADVANCED feature engineering for RMSE reduction"""
    # First apply advanced feature engineering
    engineered_df = prepare_input_with_advanced_features(input_data)
    
    # Get expected features from model if available
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        print(f"📋 Using model.feature_names_in_ for advanced model: {len(expected_features)} features")
    else:
        # Try to load from file
        try:
            if os.path.exists('models/advanced_model_info.json'):
                with open('models/advanced_model_info.json', 'r') as f:
                    model_info = json.load(f)
                    expected_features = model_info.get('feature_names', [])
                    print(f"📋 Loaded {len(expected_features)} features from advanced_model_info.json")
            else:
                expected_features = get_advanced_feature_list()
                print(f"📋 Using default feature list: {len(expected_features)} features")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            expected_features = get_advanced_feature_list()
            print(f"📋 Using default feature list: {len(expected_features)} features")
    
    print(f"📋 Model expects {len(expected_features)} features")
    print(f"📋 First 5 expected: {expected_features[:5]}")
    
    # Create a new DataFrame with exactly the expected features
    processed_df = pd.DataFrame(index=engineered_df.index)
    
    for feature in expected_features:
        if feature in engineered_df.columns:
            processed_df[feature] = engineered_df[feature]
        else:
            # Add missing column with default value 0
            processed_df[feature] = 0
            print(f"➕ Added missing column '{feature}' with value 0")
    
    # Ensure correct data types
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Fill any NaN values with 0
    processed_df = processed_df.fillna(0)
    
    # Ensure columns are in the exact order expected by the model
    if model is not None and hasattr(model, 'feature_names_in_'):
        processed_df = processed_df[list(model.feature_names_in_)]
    
    print(f"✅ Processed data shape: {processed_df.shape}")
    print(f"✅ Processed columns: {list(processed_df.columns)}")
    print(f"✅ Total advanced features: {len(processed_df.columns)}")
    
    return processed_df

def get_engineered_feature_list():
    """Return the list of ALL 38 features expected by the improved model in EXACT ORDER"""
    expected_features = [
        'Hour',
        'Temperature(°C)',
        'Humidity(%)',
        'Wind speed (m/s)',
        'Visibility (10m)',
        'Dew point temperature(°C)',
        'Solar Radiation (MJ/m2)',
        'Rainfall(mm)',
        'Snowfall (cm)',
        'hour_squared',
        'hour_cubed',
        'is_peak_hour',
        'rush_hour_intensity',
        'is_weekend',
        'day_of_week',
        'temp_hour_interaction',
        'temp_humidity_interaction',
        'wind_chill',
        'heat_index',
        'weather_severity_score',
        'comfort_index',
        'demand_lag_24',
        'demand_lag_168',
        'demand_rolling_3h',
        'demand_rolling_24h',
        'temp_deviation_from_season',
        'is_daylight',
        'workday_hour_interaction',
        'hour_category_evening',
        'hour_category_morning',
        'hour_category_night',
        'weather_category_good',
        'weather_category_perfect',
        'weather_category_poor',
        'weather_category_severe',
        'Seasons_Spring',
        'Seasons_Summer',
        'Seasons_Winter'
    ]
    
    print(f"📋 Improved model expects {len(expected_features)} features in specific order")
    return expected_features

def get_advanced_feature_list():
    """
    Return the list of features expected by the advanced model
    This should match EXACTLY what advanced_model_info.json shows
    """
    return [
        "hour",
        "temperature",
        "Humidity%",
        "Wind_speed_m/s",
        "visibility",
        "dew_point",
        "Solar_Radiation_MJ/m2",
        "rainfall",
        "snowfall",
        "hour_sin",
        "hour_cos",
        "hour_squared",
        "day_of_week",
        "is_weekend",
        "temp_hour_interaction",
        "temp_squared",
        "is_peak_hour",
        "rush_hour_intensity",
        "good_biking_conditions",
        "season_Autumn",
        "season_Spring",
        "season_Summer",
        "season_Winter",
        "holiday_Holiday",
        "holiday_No Holiday",
        "functioning_day_No",
        "functioning_day_Yes",
        "temp_category_cold",
        "temp_category_cool",
        "temp_category_freezing",
        "temp_category_hot",
        "temp_category_warm",
        "time_of_day_afternoon",
        "time_of_day_evening",
        "time_of_day_morning",
        "time_of_day_night"
    ]

def validate_input(data, model_type='original'):
    """
    Validate input data for specific model type
    
    Args:
        data: Input dictionary
        model_type: 'original', 'improved', 'advanced', or 'xgboost'
    """
    errors = []
    
    # Check for required fields with field name mapping
    required_fields = []
    
    if model_type == 'original':
        required_fields = [
            'Hour',
            'Temperature(캜)',  # Original model uses Korean character
            'Humidity(%)',
            'Wind speed (m/s)',
            'Visibility (10m)',
            'Dew point temperature(캜)',  # Original model uses Korean character
            'Solar Radiation (MJ/m2)',
            'Rainfall(mm)',
            'Snowfall (cm)'
        ]
    elif model_type == 'improved':
        # Accept EITHER format for improved model
        required_fields = [
            'Hour',
            'Humidity(%)',
            'Wind speed (m/s)',
            'Visibility (10m)',
            'Solar Radiation (MJ/m2)',
            'Rainfall(mm)',
            'Snowfall (cm)',
            'day_of_week'  # Additional required for improved model
        ]
        # Temperature can be in either format
        if 'Temperature(°C)' not in data and 'Temperature(캜)' not in data:
            errors.append('Temperature is required (use Temperature(°C) or Temperature(캜))')
        
        # Dew point can be in either format
        if 'Dew point temperature(°C)' not in data and 'Dew point temperature(캜)' not in data:
            errors.append('Dew point temperature is required (use Dew point temperature(°C) or Dew point temperature(캜))')
    
    elif model_type == 'advanced':
        # Check for required fields in any accepted format
        # Check Hour
        if not any(key in data for key in ['Hour', 'hour']):
            errors.append('Hour is required (use Hour or hour)')
        
        # Check Temperature
        if not any(key in data for key in ['Temperature(°C)', 'Temperature(캜)', 'temperature']):
            errors.append('Temperature is required (use Temperature(°C), Temperature(캜), or temperature)')
        
        # Check Humidity
        if not any(key in data for key in ['Humidity(%)', 'Humidity%', 'humidity']):
            errors.append('Humidity is required (use Humidity(%), Humidity%, or humidity)')
        
        # Check Wind Speed
        if not any(key in data for key in ['Wind speed (m/s)', 'wind_speed', 'Wind_speed_m/s']):
            errors.append('Wind speed is required (use Wind speed (m/s), wind_speed, or Wind_speed_m/s)')
        
        # Check Visibility
        if not any(key in data for key in ['Visibility (10m)', 'visibility']):
            errors.append('Visibility is required (use Visibility (10m) or visibility)')
        
        # Check Solar Radiation
        if not any(key in data for key in ['Solar Radiation (MJ/m2)', 'solar_radiation', 'Solar_Radiation_MJ/m2']):
            errors.append('Solar Radiation is required (use Solar Radiation (MJ/m2), solar_radiation, or Solar_Radiation_MJ/m2)')
        
        # Check Rainfall
        if not any(key in data for key in ['Rainfall(mm)', 'rainfall']):
            errors.append('Rainfall is required (use Rainfall(mm) or rainfall)')
        
        # Check Snowfall
        if not any(key in data for key in ['Snowfall (cm)', 'snowfall']):
            errors.append('Snowfall is required (use Snowfall (cm) or snowfall)')
        
        # Check Dew Point
        if not any(key in data for key in ['Dew point temperature(°C)', 'Dew point temperature(캜)', 'dew_point']):
            errors.append('Dew point temperature is required (use Dew point temperature(°C), Dew point temperature(캜), or dew_point)')
        
        # Check day_of_week
        if 'day_of_week' not in data:
            errors.append('day_of_week is required')
    
    elif model_type == 'xgboost':
        # Check for required fields in any accepted format for XGBoost
        # Check Hour
        if not any(key in data for key in ['Hour', 'hour']):
            errors.append('Hour is required (use Hour or hour)')
        
        # Check Temperature
        if not any(key in data for key in ['Temperature(°C)', 'Temperature(캜)', 'temperature']):
            errors.append('Temperature is required (use Temperature(°C), Temperature(캜), or temperature)')
        
        # Check Humidity
        if not any(key in data for key in ['Humidity(%)', 'humidity']):
            errors.append('Humidity is required (use Humidity(%) or humidity)')
        
        # Check Wind Speed
        if not any(key in data for key in ['Wind speed (m/s)', 'wind_speed']):
            errors.append('Wind speed is required (use Wind speed (m/s) or wind_speed)')
        
        # Check Visibility
        if not any(key in data for key in ['Visibility (10m)', 'visibility']):
            errors.append('Visibility is required (use Visibility (10m) or visibility)')
        
        # Check Solar Radiation
        if not any(key in data for key in ['Solar Radiation (MJ/m2)', 'solar_radiation']):
            errors.append('Solar Radiation is required (use Solar Radiation (MJ/m2) or solar_radiation)')
        
        # Check Rainfall
        if not any(key in data for key in ['Rainfall(mm)', 'rainfall']):
            errors.append('Rainfall is required (use Rainfall(mm) or rainfall)')
        
        # Check Snowfall
        if not any(key in data for key in ['Snowfall (cm)', 'snowfall']):
            errors.append('Snowfall is required (use Snowfall (cm) or snowfall)')
        
        # Check Dew Point
        if not any(key in data for key in ['Dew point temperature(°C)', 'Dew point temperature(캜)', 'dew_point']):
            errors.append('Dew point temperature is required (use Dew point temperature(°C), Dew point temperature(캜), or dew_point)')
        
        # Check day_of_week
        if 'day_of_week' not in data:
            errors.append('day_of_week is required')
    
    # Check other required fields for original and improved models
    if model_type in ['original', 'improved']:
        for field in required_fields:
            if field not in data:
                errors.append(f'{field} is required')
    
    # Common validations for all models
    # Validate Hour
    hour_value = None
    if 'Hour' in data:
        hour_value = data['Hour']
    elif 'hour' in data and model_type in ['advanced', 'xgboost']:
        hour_value = data['hour']
    
    if hour_value is not None:
        try:
            hour = float(hour_value)
            if not (0 <= hour <= 23):
                errors.append('Hour must be between 0 and 23')
        except (ValueError, TypeError):
            errors.append('Hour must be a number')
    
    # Validate Temperature
    temp_value = None
    if 'Temperature(°C)' in data:
        temp_value = data['Temperature(°C)']
    elif 'Temperature(캜)' in data:
        temp_value = data['Temperature(캜)']
    elif 'temperature' in data and model_type in ['advanced', 'xgboost']:
        temp_value = data['temperature']
    
    if temp_value is not None:
        try:
            temp = float(temp_value)
            if not (-20 <= temp <= 50):
                errors.append('Temperature must be between -20°C and 50°C')
        except (ValueError, TypeError):
            errors.append('Temperature must be a number')
    
    # Validate Humidity
    humidity_value = None
    if 'Humidity(%)' in data:
        humidity_value = data['Humidity(%)']
    elif 'Humidity%' in data and model_type == 'advanced':
        humidity_value = data['Humidity%']
    elif 'humidity' in data and model_type == 'xgboost':
        humidity_value = data['humidity']
    
    if humidity_value is not None:
        try:
            humidity = float(humidity_value)
            if not (0 <= humidity <= 100):
                errors.append('Humidity must be between 0% and 100%')
        except (ValueError, TypeError):
            errors.append('Humidity must be a number')
    
    # Validate Wind Speed
    wind_value = None
    if 'Wind speed (m/s)' in data:
        wind_value = data['Wind speed (m/s)']
    elif 'wind_speed' in data and model_type == 'xgboost':
        wind_value = data['wind_speed']
    elif 'Wind_speed_m/s' in data and model_type == 'advanced':
        wind_value = data['Wind_speed_m/s']
    
    if wind_value is not None:
        try:
            wind = float(wind_value)
            if wind < 0 or wind > 50:
                errors.append('Wind speed must be between 0 and 50 m/s')
        except (ValueError, TypeError):
            errors.append('Wind speed must be a number')
    
    # Validate Visibility
    visibility_value = None
    if 'Visibility (10m)' in data:
        visibility_value = data['Visibility (10m)']
    elif 'visibility' in data and model_type in ['advanced', 'xgboost']:
        visibility_value = data['visibility']
    
    if visibility_value is not None:
        try:
            visibility = float(visibility_value)
            if visibility < 0 or visibility > 5000:
                errors.append('Visibility must be between 0 and 5000 (10m)')
        except (ValueError, TypeError):
            errors.append('Visibility must be a number')
    
    # Validate Solar Radiation
    solar_value = None
    if 'Solar Radiation (MJ/m2)' in data:
        solar_value = data['Solar Radiation (MJ/m2)']
    elif 'solar_radiation' in data and model_type == 'xgboost':
        solar_value = data['solar_radiation']
    elif 'Solar_Radiation_MJ/m2' in data and model_type == 'advanced':
        solar_value = data['Solar_Radiation_MJ/m2']
    
    if solar_value is not None:
        try:
            solar = float(solar_value)
            if solar < 0 or solar > 5:
                errors.append('Solar Radiation must be between 0 and 5 MJ/m2')
        except (ValueError, TypeError):
            errors.append('Solar Radiation must be a number')
    
    # Validate Rainfall
    rainfall_value = None
    if 'Rainfall(mm)' in data:
        rainfall_value = data['Rainfall(mm)']
    elif 'rainfall' in data and model_type in ['advanced', 'xgboost']:
        rainfall_value = data['rainfall']
    
    if rainfall_value is not None:
        try:
            rainfall = float(rainfall_value)
            if rainfall < 0 or rainfall > 100:
                errors.append('Rainfall must be between 0 and 100 mm')
        except (ValueError, TypeError):
            errors.append('Rainfall must be a number')
    
    # Validate Snowfall
    snowfall_value = None
    if 'Snowfall (cm)' in data:
        snowfall_value = data['Snowfall (cm)']
    elif 'snowfall' in data and model_type in ['advanced', 'xgboost']:
        snowfall_value = data['snowfall']
    
    if snowfall_value is not None:
        try:
            snowfall = float(snowfall_value)
            if snowfall < 0 or snowfall > 50:
                errors.append('Snowfall must be between 0 and 50 cm')
        except (ValueError, TypeError):
            errors.append('Snowfall must be a number')
    
    # Validate Dew Point
    dew_point_value = None
    if 'Dew point temperature(°C)' in data:
        dew_point_value = data['Dew point temperature(°C)']
    elif 'Dew point temperature(캜)' in data:
        dew_point_value = data['Dew point temperature(캜)']
    elif 'dew_point' in data and model_type in ['advanced', 'xgboost']:
        dew_point_value = data['dew_point']
    
    if dew_point_value is not None:
        try:
            dew_point = float(dew_point_value)
            if dew_point < -30 or dew_point > 30:
                errors.append('Dew point temperature must be between -30°C and 30°C')
        except (ValueError, TypeError):
            errors.append('Dew point temperature must be a number')
    
    # Validate day_of_week for models that need it
    if model_type in ['improved', 'advanced', 'xgboost']:
        day_value = None
        if 'day_of_week' in data:
            day_value = data['day_of_week']
        
        if day_value is not None:
            try:
                # Handle string input from form
                day_str = str(day_value)
                day = int(float(day_str))  # Convert string like "0" or float like 0.0 to int
                if not (0 <= day <= 6):
                    errors.append('day_of_week must be between 0 (Monday) and 6 (Sunday)')
            except (TypeError, ValueError):
                errors.append('day_of_week must be an integer between 0 and 6')
        else:
            errors.append('day_of_week is required for this model')
    
    return errors

def save_prediction_to_history(input_data, prediction_result):
    """Save prediction to history file"""
    try:
        history_file = 'data/predictions_history.json'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        # Load existing history
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Create history entry
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'prediction': prediction_result.get('prediction', 0),
            'rounded': prediction_result.get('rounded', 0),
            'confidence': prediction_result.get('confidence', 0),
            'model_type': prediction_result.get('model_type', 'original'),
            'success': prediction_result.get('success', False)
        }
        
        # Add to history
        history.append(history_entry)
        
        # Keep only last 100 predictions
        if len(history) > 100:
            history = history[-100:]
        
        # Save back to file
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✅ Prediction saved to history. Total predictions: {len(history)}")
        
    except Exception as e:
        print(f"❌ Error saving prediction to history: {e}")

def get_prediction_history(limit=50):
    """Get prediction history"""
    try:
        history_file = 'data/predictions_history.json'
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            # Return limited number of predictions
            return history[-limit:]
        else:
            return []
        
    except Exception as e:
        print(f"❌ Error loading prediction history: {e}")
        return []

def get_model_feature_names(model_type='original'):
    """Get feature names for a specific model type"""
    if model_type == 'original':
        # Return what the original model actually expects
        return [
            'Hour',
            'Temperature(캜)',  # ORIGINAL: This model was trained with (캜)
            'Humidity(%)',
            'Wind speed (m/s)',
            'Visibility (10m)',
            'Dew point temperature(캜)',  # ORIGINAL: This model was trained with (캜)
            'Solar Radiation (MJ/m2)',
            'Rainfall(mm)',
            'Snowfall (cm)',
            'Seasons_Spring',
            'Seasons_Summer',
            'Seasons_Winter',
            'Holiday_No Holiday',
            'Functioning Day_Yes'
        ]
    elif model_type == 'improved':
        return get_engineered_feature_list()
    elif model_type == 'advanced' or model_type == 'xgboost':
        return get_advanced_feature_list()
    elif model_type == 'unified' or model_type == 'ensemble':
        return load_unified_feature_names()
    else:
        return []


# ============================================================================
# UNIFIED PIPELINE SUPPORT
# ============================================================================

_unified_cache = {}

def load_unified_pipeline():
    """Load models and artifacts from the unified pipeline.
    Returns dict with model, scaler, feature_names, config, or None if not found.
    """
    if _unified_cache.get('loaded'):
        return _unified_cache

    base = os.path.join(os.getcwd(), 'models', 'unified_pipeline')
    if not os.path.isdir(base):
        base = os.path.join(os.path.dirname(__file__), '..', 'models', 'unified_pipeline')
    if not os.path.isdir(base):
        logger.warning("Unified pipeline directory not found")
        return None

    try:
        config_path = os.path.join(base, 'pipeline_config.json')
        with open(config_path) as f:
            config = json.load(f)

        feature_path = os.path.join(base, 'feature_names.json')
        with open(feature_path) as f:
            feature_names = json.load(f)

        scaler = joblib.load(os.path.join(base, 'scaler.pkl'))

        # Load best model
        best_model_name = config.get('best_model', 'stacking_ensemble')
        model_file = os.path.join(base, f'{best_model_name}.pkl')
        if not os.path.exists(model_file):
            model_file = os.path.join(base, 'stacking_ensemble.pkl')
        model = joblib.load(model_file)

        # Load demand cache for lag features
        demand_cache = []
        cache_path = os.path.join(base, 'demand_cache.json')
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                demand_cache = json.load(f).get('last_168_values', [])

        # Load lag medians
        lag_medians = {}
        lag_path = os.path.join(base, 'lag_medians.json')
        if os.path.exists(lag_path):
            with open(lag_path) as f:
                lag_medians = json.load(f)

        _unified_cache.update({
            'loaded': True,
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'config': config,
            'demand_cache': demand_cache,
            'lag_medians': lag_medians,
            'model_name': best_model_name,
            'base_dir': base,
        })

        logger.info(f"Loaded unified pipeline: {best_model_name} with {len(feature_names)} features")
        return _unified_cache

    except Exception as e:
        logger.error(f"Failed to load unified pipeline: {e}")
        return None


def load_unified_feature_names():
    """Return feature names from unified pipeline, or empty list."""
    pipeline = load_unified_pipeline()
    if pipeline:
        return pipeline['feature_names']
    return []


def preprocess_unified(input_data: dict) -> np.ndarray:
    """Preprocess input data for the unified pipeline model.
    Builds a feature vector matching the training feature order.
    """
    pipeline = load_unified_pipeline()
    if not pipeline:
        raise ValueError("Unified pipeline not loaded")

    feature_names = pipeline['feature_names']
    scaler = pipeline['scaler']
    lag_medians = pipeline['lag_medians']
    demand_cache = pipeline['demand_cache']
    config = pipeline['config']
    scale_cols = config.get('scale_cols', [])

    # Map input to standardized names
    hour = float(input_data.get('Hour', input_data.get('hour', 12)))
    temp = float(input_data.get('Temperature(°C)', input_data.get('temperature', 20)))
    humidity = float(input_data.get('Humidity(%)', input_data.get('humidity', 50)))
    wind = float(input_data.get('Wind speed (m/s)', input_data.get('wind_speed', 2.5)))
    vis = float(input_data.get('Visibility (10m)', input_data.get('visibility', 2000)))
    dew = float(input_data.get('Dew point temperature(°C)', input_data.get('dew_point', 10)))
    solar = float(input_data.get('Solar Radiation (MJ/m2)', input_data.get('solar_radiation', 1.5)))
    rain = float(input_data.get('Rainfall(mm)', input_data.get('rainfall', 0)))
    snow = float(input_data.get('Snowfall (cm)', input_data.get('snowfall', 0)))

    # Season
    season_spring = float(input_data.get('Seasons_Spring', input_data.get('season_Spring', 0)))
    season_summer = float(input_data.get('Seasons_Summer', input_data.get('season_Summer', 0)))
    season_winter = float(input_data.get('Seasons_Winter', input_data.get('season_Winter', 0)))
    season_autumn = float(input_data.get('Seasons_Autumn', input_data.get('season_Autumn', 0)))
    if season_spring == 0 and season_summer == 0 and season_winter == 0 and season_autumn == 0:
        season_summer = 1

    # Holiday / Functioning
    is_holiday = float(input_data.get('is_holiday', 0))
    if 'Holiday_No Holiday' in input_data:
        is_holiday = 1.0 - float(input_data['Holiday_No Holiday'])
    is_functioning = float(input_data.get('is_functioning_day', input_data.get('Functioning Day_Yes', 1)))

    day_of_week = int(input_data.get('day_of_week', datetime.now().weekday()))

    # Build feature dict
    feat = {}
    feat['Hour'] = hour
    feat['Temperature(°C)'] = temp
    feat['Humidity(%)'] = humidity
    feat['Wind speed (m/s)'] = wind
    feat['Visibility (10m)'] = vis
    feat['Dew point temperature(°C)'] = dew
    feat['Solar Radiation (MJ/m2)'] = solar
    feat['Rainfall(mm)'] = rain
    feat['Snowfall (cm)'] = snow

    # Time features
    feat['hour_sin'] = float(np.sin(2 * np.pi * hour / 24))
    feat['hour_cos'] = float(np.cos(2 * np.pi * hour / 24))
    feat['hour_squared'] = hour ** 2
    month = int(input_data.get('month', datetime.now().month))
    feat['month'] = month
    feat['month_sin'] = float(np.sin(2 * np.pi * month / 12))
    feat['month_cos'] = float(np.cos(2 * np.pi * month / 12))
    feat['day_of_week'] = day_of_week
    feat['dow_sin'] = float(np.sin(2 * np.pi * day_of_week / 7))
    feat['dow_cos'] = float(np.cos(2 * np.pi * day_of_week / 7))
    feat['is_weekend'] = 1 if day_of_week >= 5 else 0

    feat['is_peak_hour'] = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    feat['rush_hour_intensity'] = (
        1.0 if hour in (7, 8, 17, 18)
        else 0.6 if hour in (6, 9, 16, 19)
        else 0.3 if hour in (10, 15, 20)
        else 0.0
    )

    feat['temp_hour'] = temp * hour
    feat['temp_humidity'] = temp * humidity
    feat['comfort_index'] = temp - humidity / 10
    feat['has_precipitation'] = 1 if rain > 0 or snow > 0 else 0

    # Lag features from demand cache or medians
    if demand_cache and len(demand_cache) >= 168:
        feat['demand_lag_24h'] = demand_cache[-24] if len(demand_cache) >= 24 else lag_medians.get('demand_lag_24h', 500)
        feat['demand_lag_168h'] = demand_cache[-168]
        feat['demand_rolling_3h_mean'] = float(np.mean(demand_cache[-4:-1])) if len(demand_cache) >= 4 else lag_medians.get('demand_rolling_3h_mean', 500)
        feat['demand_rolling_24h_mean'] = float(np.mean(demand_cache[-25:-1])) if len(demand_cache) >= 25 else lag_medians.get('demand_rolling_24h_mean', 500)
    else:
        for lag_col in ['demand_lag_24h', 'demand_lag_168h', 'demand_rolling_3h_mean', 'demand_rolling_24h_mean']:
            feat[lag_col] = lag_medians.get(lag_col, 500)

    # Season dummies
    for name in ['season_Autumn', 'season_Spring', 'season_Summer', 'season_Winter']:
        if name == 'season_Spring':
            feat[name] = season_spring
        elif name == 'season_Summer':
            feat[name] = season_summer
        elif name == 'season_Winter':
            feat[name] = season_winter
        elif name == 'season_Autumn':
            feat[name] = season_autumn

    feat['is_holiday'] = is_holiday
    feat['is_functioning_day'] = is_functioning

    # Build array in correct feature order
    row = []
    for fn in feature_names:
        row.append(feat.get(fn, 0.0))

    row_array = np.array([row])

    # Apply scaling to correct columns
    row_df = pd.DataFrame(row_array, columns=feature_names)
    if scale_cols:
        cols_to_scale = [c for c in scale_cols if c in feature_names]
        if cols_to_scale:
            col_indices = [feature_names.index(c) for c in cols_to_scale]
            row_df[cols_to_scale] = scaler.transform(row_df[cols_to_scale])

    return row_df.values