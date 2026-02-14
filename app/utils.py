"""
app/utils.py - Complete updated version with advanced feature engineering
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
    Includes all 5 requested features plus additional engineered features
    """
    # Convert to DataFrame if dict
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data
    
    # Ensure required numeric fields
    numeric_fields = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 
                     'Visibility (10m)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 
                     'Snowfall (cm)', 'Dew point temperature(°C)']
    
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
    
    # Handle day_of_week conversion (FIX for string input)
    if 'day_of_week' in df.columns:
        try:
            if isinstance(df['day_of_week'].iloc[0], str):
                df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype(int)
            else:
                df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype(int)
        except Exception as e:
            print(f"⚠️ Error converting day_of_week: {e}")
            df['day_of_week'] = 0
    
    # ==================================================
    # 1. HOUR SQUARED FEATURE (hour²)
    # ==================================================
    df['Hour_squared'] = df['Hour'] ** 2
    
    # ==================================================
    # 2. TEMPERATURE-HOUR INTERACTION TERM
    # ==================================================
    df['Temp_Hour_interaction'] = df['Temperature(°C)'] * df['Hour']
    
    # ==================================================
    # 3. WEEKDAY/WEEKEND FLAG
    # ==================================================
    if 'day_of_week' in df.columns:
        df['Day_of_week'] = df['day_of_week']
        # Ensure it's integer for comparison
        try:
            day_val = int(df['Day_of_week'].iloc[0])
        except (ValueError, TypeError):
            day_val = 0
        # FIXED: Convert boolean to integer properly
        df['Is_weekend'] = 1 if day_val >= 5 else 0  # 5=Saturday, 6=Sunday
    else:
        # Default to weekday if not provided
        df['Is_weekend'] = 0
    
    # ==================================================
    # 4. PEAK HOURS INDICATOR (7-9 AM, 5-7 PM)
    # ==================================================
    df['Is_morning_peak'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)).astype(int)
    df['Is_evening_peak'] = ((df['Hour'] >= 17) & (df['Hour'] <= 19)).astype(int)
    df['Is_peak_hour'] = (df['Is_morning_peak'] | df['Is_evening_peak']).astype(int)
    
    # ==================================================
    # 5. WEATHER CONDITIONS CATEGORIZATION
    # ==================================================
    
    # Temperature categories
    def get_temp_category(temp):
        if temp < 0:
            return 'Freezing'
        elif temp < 10:
            return 'Cold'
        elif temp < 20:
            return 'Cool'
        elif temp < 30:
            return 'Warm'
        else:
            return 'Hot'
    
    # Precipitation categories
    def get_precip_category(rainfall, snowfall):
        if rainfall > 5 or snowfall > 2:
            return 'Heavy'
        elif rainfall > 0.5 or snowfall > 0:
            return 'Light'
        else:
            return 'None'
    
    # Wind categories
    def get_wind_category(wind_speed):
        if wind_speed > 7:
            return 'Strong'
        elif wind_speed > 3:
            return 'Moderate'
        else:
            return 'Calm'
    
    # Apply categorizations
    temp_cat = df['Temperature(°C)'].apply(get_temp_category).iloc[0]
    precip_cat = get_precip_category(
        df['Rainfall(mm)'].iloc[0] if 'Rainfall(mm)' in df.columns else 0,
        df['Snowfall (cm)'].iloc[0] if 'Snowfall (cm)' in df.columns else 0
    )
    wind_cat = get_wind_category(
        df['Wind speed (m/s)'].iloc[0] if 'Wind speed (m/s)' in df.columns else 0
    )
    
    weather_category = f"{temp_cat}_{precip_cat}_{wind_cat}"
    
    # Create all possible weather category columns
    possible_categories = [
        'Freezing_None_Calm', 'Freezing_None_Moderate', 'Freezing_None_Strong',
        'Freezing_Light_Calm', 'Freezing_Light_Moderate', 'Freezing_Light_Strong',
        'Freezing_Heavy_Calm', 'Freezing_Heavy_Moderate', 'Freezing_Heavy_Strong',
        'Cold_None_Calm', 'Cold_None_Moderate', 'Cold_None_Strong',
        'Cold_Light_Calm', 'Cold_Light_Moderate', 'Cold_Light_Strong',
        'Cold_Heavy_Calm', 'Cold_Heavy_Moderate', 'Cold_Heavy_Strong',
        'Cool_None_Calm', 'Cool_None_Moderate', 'Cool_None_Strong',
        'Cool_Light_Calm', 'Cool_Light_Moderate', 'Cool_Light_Strong',
        'Cool_Heavy_Calm', 'Cool_Heavy_Moderate', 'Cool_Heavy_Strong',
        'Warm_None_Calm', 'Warm_None_Moderate', 'Warm_None_Strong',
        'Warm_Light_Calm', 'Warm_Light_Moderate', 'Warm_Light_Strong',
        'Warm_Heavy_Calm', 'Warm_Heavy_Moderate', 'Warm_Heavy_Strong',
        'Hot_None_Calm', 'Hot_None_Moderate', 'Hot_None_Strong',
        'Hot_Light_Calm', 'Hot_Light_Moderate', 'Hot_Light_Strong',
        'Hot_Heavy_Calm', 'Hot_Heavy_Moderate', 'Hot_Heavy_Strong'
    ]
    
    for category in possible_categories:
        df[f'Weather_{category}'] = 0
    
    # Set the actual category to 1
    df[f'Weather_{weather_category}'] = 1
    
    # ==================================================
    # ADDITIONAL ADVANCED FEATURES FOR RMSE REDUCTION
    # ==================================================
    
    # 6. Comfort index: temperature and humidity combination
    df['Comfort_index'] = df['Temperature(°C)'] - (df['Humidity(%)'] / 10)
    
    # 7. Visibility normalized
    df['Visibility_normalized'] = df['Visibility (10m)'] / 2000
    
    # 8. Solar radiation categories
    df['Solar_high'] = (df['Solar Radiation (MJ/m2)'] > 1.0).astype(int)
    
    # 9. Time of day categories
    hour = df['Hour'].iloc[0]
    time_categories = ['Time_Night', 'Time_Morning', 'Time_Afternoon', 'Time_Evening']
    for cat in time_categories:
        df[cat] = 0
    
    if hour < 6:
        df['Time_Night'] = 1
    elif hour < 12:
        df['Time_Morning'] = 1
    elif hour < 18:
        df['Time_Afternoon'] = 1
    else:
        df['Time_Evening'] = 1
    
    # 10. Temperature squared (non-linear effect)
    df['Temp_squared'] = df['Temperature(°C)'] ** 2
    
    # 11. Humidity squared
    df['Humidity_squared'] = df['Humidity(%)'] ** 2
    
    # 12. Temperature-Humidity-Wind interaction
    df['Temp_Humidity_Wind'] = df['Temperature(°C)'] * df['Humidity(%)'] * df['Wind speed (m/s)']
    
    # 13. Rush hour intensity (weighted)
    def rush_hour_intensity_advanced(hour):
        if hour in [7, 8, 17, 18]:  # Peak of peaks
            return 1.0
        elif hour in [6, 9, 16, 19]:  # Shoulder peaks
            return 0.7
        elif hour in [10, 15, 20]:  # Near peaks
            return 0.3
        else:
            return 0.0
    
    df['Rush_hour_intensity'] = df['Hour'].apply(rush_hour_intensity_advanced)
    
    # 14. Weather severity score (comprehensive)
    def weather_severity_score_advanced(row):
        score = 0
        
        # Temperature penalty
        temp = row['Temperature(°C)']
        if temp < 0:
            score += 5
        elif temp < 5:
            score += 3
        elif temp < 10:
            score += 2
        elif temp > 30:
            score += 4
        elif temp > 25:
            score += 2
        
        # Precipitation penalty
        if row.get('Rainfall(mm)', 0) > 0:
            score += min(5, row['Rainfall(mm)'] * 2)
        
        if row.get('Snowfall (cm)', 0) > 0:
            score += min(5, row['Snowfall (cm)'] * 3)
        
        # Wind penalty
        if row['Wind speed (m/s)'] > 10:
            score += 5
        elif row['Wind speed (m/s)'] > 7:
            score += 3
        elif row['Wind speed (m/s)'] > 5:
            score += 2
        
        # Visibility penalty
        if row['Visibility (10m)'] < 500:
            score += 4
        elif row['Visibility (10m)'] < 1000:
            score += 2
        
        return score
    
    df['Weather_severity_score'] = df.apply(weather_severity_score_advanced, axis=1)
    
    # 15. Good biking conditions indicator
    df['Good_biking_conditions'] = (
        (df['Temperature(°C)'] >= 15) & 
        (df['Temperature(°C)'] <= 25) & 
        (df['Rainfall(mm)'] == 0) & 
        (df['Snowfall (cm)'] == 0) & 
        (df['Wind speed (m/s)'] < 5)
    ).astype(int)
    
    # ==================================================
    # ORIGINAL FEATURES (keep these)
    # ==================================================
    
    # Handle season encoding
    season_cols = ['Seasons_Spring', 'Seasons_Summer', 'Seasons_Winter', 'Seasons_Autumn']
    for col in season_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure holiday encoding
    if 'Holiday_No Holiday' not in df.columns:
        df['Holiday_No Holiday'] = 1
        df['Holiday_Holiday'] = 0
    
    # Ensure functioning day encoding
    if 'Functioning Day_Yes' not in df.columns:
        df['Functioning Day_Yes'] = 1
        df['Functioning Day_No'] = 0
    
    # Handle field name mappings for backward compatibility
    if 'Temperature(캜)' in df.columns:
        df['Temperature(°C)'] = df['Temperature(캜)']
    
    if 'Dew point temperature(캜)' in df.columns:
        df['Dew point temperature(°C)'] = df['Dew point temperature(캜)']
    
    return df

def preprocess_with_advanced_features(input_data, model=None):
    """Preprocess with ADVANCED feature engineering for RMSE reduction"""
    # First apply advanced feature engineering
    engineered_df = prepare_input_with_advanced_features(input_data)
    
    # Get expected features from model if available, otherwise use default list
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        print(f"📋 Using model.feature_names_in_ for advanced model: {len(expected_features)} features")
    else:
        expected_features = get_advanced_feature_list()
        print(f"📋 Using default feature list for advanced model: {len(expected_features)} features")
    
    print(f"📋 Advanced model expects {len(expected_features)} features")
    
    # Create a new DataFrame with exactly the expected features
    processed_df = pd.DataFrame()
    
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
    This should match EXACTLY what train_with_advanced_features.py produces
    """
    # Base features (from original dataset)
    base_features = [
        'Hour',
        'Temperature(°C)',
        'Humidity(%)',
        'Wind speed (m/s)',
        'Visibility (10m)',
        'Dew point temperature(°C)',
        'Solar Radiation (MJ/m2)',
        'Rainfall(mm)',
        'Snowfall (cm)',
        'Seasons_Spring',
        'Seasons_Summer',
        'Seasons_Winter',
        'Holiday_No Holiday',
        'Functioning Day_Yes'
    ]
    
    # Advanced engineered features
    advanced_features = [
        # Requested features
        'Hour_squared',
        'Temp_Hour_interaction',
        'Is_weekend',
        'Is_morning_peak',
        'Is_evening_peak',
        'Is_peak_hour',
        
        # Additional advanced features
        'Comfort_index',
        'Visibility_normalized',
        'Solar_high',
        'Temp_squared',
        'Humidity_squared',
        'Temp_Humidity_Wind',
        'Rush_hour_intensity',
        'Weather_severity_score',
        'Good_biking_conditions'
    ]
    
    # Time categories
    time_features = [
        'Time_Night',
        'Time_Morning', 
        'Time_Afternoon',
        'Time_Evening'
    ]
    
    # All features combined
    all_features = base_features + advanced_features + time_features
    
    # Add weather categories (45 possible combinations)
    weather_base = ['Freezing', 'Cold', 'Cool', 'Warm', 'Hot']
    precip = ['None', 'Light', 'Heavy']
    wind = ['Calm', 'Moderate', 'Strong']
    
    for temp in weather_base:
        for p in precip:
            for w in wind:
                all_features.append(f'Weather_{temp}_{p}_{w}')
    
    return all_features

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
    elif model_type == 'advanced' or model_type == 'xgboost':
        required_fields = [
            'Hour',
            'Humidity(%)',
            'Wind speed (m/s)',
            'Visibility (10m)',
            'Solar Radiation (MJ/m2)',
            'Rainfall(mm)',
            'Snowfall (cm)',
            'day_of_week'
        ]
        # Advanced models accept either temperature format
        if 'Temperature(°C)' not in data and 'Temperature(캜)' not in data:
            errors.append('Temperature is required (use Temperature(°C) or Temperature(캜))')
        
        # Advanced models accept either dew point format
        if 'Dew point temperature(°C)' not in data and 'Dew point temperature(캜)' not in data:
            errors.append('Dew point temperature is required (use Dew point temperature(°C) or Dew point temperature(캜))')
    
    # Check other required fields
    for field in required_fields:
        if field not in data:
            errors.append(f'{field} is required')
    
    # Common validations for all models
    if 'Hour' in data:
        try:
            hour = float(data.get('Hour', -1))
            if not (0 <= hour <= 23):
                errors.append('Hour must be between 0 and 23')
        except (ValueError, TypeError):
            errors.append('Hour must be a number')
    
    # Check temperature (accept any format)
    if 'Temperature(°C)' in data or 'Temperature(캜)' in data:
        try:
            temp = float(data.get('Temperature(°C)', data.get('Temperature(캜)', -100)))
            if not (-20 <= temp <= 50):
                errors.append('Temperature must be between -20°C and 50°C')
        except (ValueError, TypeError):
            errors.append('Temperature must be a number')
    
    if 'Humidity(%)' in data:
        try:
            humidity = float(data.get('Humidity(%)', -1))
            if not (0 <= humidity <= 100):
                errors.append('Humidity must be between 0% and 100%')
        except (ValueError, TypeError):
            errors.append('Humidity must be a number')
    
    # Validate day_of_week for models that need it
    if model_type in ['improved', 'advanced', 'xgboost'] and 'day_of_week' in data:
        try:
            # Handle string input from form
            day_str = str(data.get('day_of_week', '0'))
            day = int(float(day_str))  # Convert string like "0" or float like 0.0 to int
            if not (0 <= day <= 6):
                errors.append('day_of_week must be between 0 (Monday) and 6 (Sunday)')
        except (TypeError, ValueError):
            errors.append('day_of_week must be an integer between 0 and 6')
    
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
    else:
        return []