import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
import json
import traceback
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath='data/raw/seoul_bike.csv'):
    """Load and preprocess data with advanced feature engineering"""
    print("📊 Loading and preprocessing data with advanced features...")
    
    # Load data with correct encoding
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        print(f"   Original shape: {df.shape}")
        print(f"   Original columns: {df.columns.tolist()}")
        
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except:
            df = pd.read_csv(filepath, encoding='cp1252')
    
    print(f"\n   First 3 rows:")
    print(df.head(3).T)
    
    # ==================================================
    # FIXED: Standardize column names
    # ==================================================
    
    # Create a clean copy with standardized names
    df_clean = df.copy()
    
    # Standardize column names (remove special characters, make lowercase)
    df_clean.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('°', '').replace('.', '') for col in df_clean.columns]
    print(f"\n   Cleaned column names: {df_clean.columns.tolist()}")
    
    # Map to expected names
    column_mapping = {
        'Hour': 'hour',
        'TemperatureC': 'temperature',
        'Temperature캜': 'temperature',
        'Humidity': 'humidity',
        'Wind_speed_ms': 'wind_speed',
        'Visibility_10m': 'visibility',
        'Dew_point_temperatureC': 'dew_point',
        'Dew_point_temperature캜': 'dew_point',
        'Solar_Radiation_MJm2': 'solar_radiation',
        'Rainfallmm': 'rainfall',
        'Snowfall_cm': 'snowfall',
        'Seasons': 'season',
        'Holiday': 'holiday',
        'Functioning_Day': 'functioning_day',
        'Rented_Bike_Count': 'bike_count',
        'Date': 'date'
    }
    
    # Apply mapping
    for old_name, new_name in column_mapping.items():
        if old_name in df_clean.columns:
            df_clean = df_clean.rename(columns={old_name: new_name})
    
    print(f"\n   Mapped columns: {df_clean.columns.tolist()}")
    
    # Check for target column
    if 'bike_count' not in df_clean.columns:
        # Try to find target column
        for col in df_clean.columns:
            if 'bike' in col.lower() or 'count' in col.lower() or 'rented' in col.lower():
                df_clean = df_clean.rename(columns={col: 'bike_count'})
                print(f"   Found and renamed target: {col} -> bike_count")
                break
    
    # ==================================================
    # FIXED: Date handling
    # ==================================================
    
    if 'date' in df_clean.columns:
        try:
            df_clean['date'] = pd.to_datetime(df_clean['date'], dayfirst=True)
            print(f"   Successfully parsed dates")
            print(f"   Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
        except Exception as e:
            print(f"   ⚠️ Error parsing dates: {e}")
            df_clean['date'] = pd.date_range(start='2017-01-01', periods=len(df_clean), freq='H')
    else:
        print("   Creating synthetic dates...")
        df_clean['date'] = pd.date_range(start='2017-01-01', periods=len(df_clean), freq='H')
    
    # Extract hour if not present
    if 'hour' not in df_clean.columns:
        print("   Extracting hour from date...")
        df_clean['hour'] = df_clean['date'].dt.hour
    
    # ==================================================
    # FIXED: Handle missing values
    # ==================================================
    
    print(f"\n   Checking for missing values...")
    missing_values = df_clean.isnull().sum()
    if missing_values.sum() > 0:
        print(f"   Found missing values:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"     {col}: {count} missing")
        # Fill missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        print("   Filled missing values with median")
    
    # ==================================================
    # FIXED: Convert categorical columns to proper types
    # ==================================================
    
    print("\n   Processing categorical columns...")
    categorical_cols = ['season', 'holiday', 'functioning_day']
    
    for col in categorical_cols:
        if col in df_clean.columns:
            # Convert to string and strip whitespace
            df_clean[col] = df_clean[col].astype(str).str.strip()
            print(f"   Processed {col}: {df_clean[col].unique()}")
    
    # ==================================================
    # ADVANCED FEATURE ENGINEERING
    # ==================================================
    
    print("\n   Starting advanced feature engineering...")
    
    # 1. Time-based features
    df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['hour'] / 24)
    df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['hour'] / 24)
    df_clean['hour_squared'] = df_clean['hour'] ** 2
    
    # 2. Date-based features
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
    df_clean['is_weekend'] = (df_clean['day_of_week'] >= 5).astype(int)
    
    # 3. Temperature-based features
    if 'temperature' in df_clean.columns:
        df_clean['temp_hour_interaction'] = df_clean['temperature'] * df_clean['hour']
        df_clean['temp_squared'] = df_clean['temperature'] ** 2
        
        # Temperature categories
        conditions = [
            (df_clean['temperature'] < 0),
            (df_clean['temperature'] < 10),
            (df_clean['temperature'] < 20),
            (df_clean['temperature'] < 30),
            (df_clean['temperature'] >= 30)
        ]
        choices = ['freezing', 'cold', 'cool', 'warm', 'hot']
        df_clean['temp_category'] = np.select(conditions, choices, default='warm')
    
    # 4. Peak hour indicators
    df_clean['is_morning_peak'] = ((df_clean['hour'] >= 7) & (df_clean['hour'] <= 9)).astype(int)
    df_clean['is_evening_peak'] = ((df_clean['hour'] >= 17) & (df_clean['hour'] <= 19)).astype(int)
    df_clean['is_peak_hour'] = (df_clean['is_morning_peak'] | df_clean['is_evening_peak']).astype(int)
    
    # 5. Weather comfort features
    if 'temperature' in df_clean.columns and 'humidity' in df_clean.columns:
        df_clean['comfort_index'] = df_clean['temperature'] - (df_clean['humidity'] / 10)
    
    if 'wind_speed' in df_clean.columns:
        df_clean['wind_speed_squared'] = df_clean['wind_speed'] ** 2
    
    if 'humidity' in df_clean.columns:
        df_clean['humidity_squared'] = df_clean['humidity'] ** 2
    
    # 6. Rush hour intensity
    def rush_hour_intensity(hour):
        if hour in [7, 8, 17, 18]:
            return 1.0
        elif hour in [6, 9, 16, 19]:
            return 0.6
        elif hour in [10, 15, 20]:
            return 0.3
        else:
            return 0.0
    
    df_clean['rush_hour_intensity'] = df_clean['hour'].apply(rush_hour_intensity)
    
    # 7. Good biking conditions
    conditions_list = []
    if 'temperature' in df_clean.columns:
        conditions_list.append((df_clean['temperature'] >= 10) & (df_clean['temperature'] <= 30))
    if 'rainfall' in df_clean.columns:
        conditions_list.append(df_clean['rainfall'] == 0)
    if 'snowfall' in df_clean.columns:
        conditions_list.append(df_clean['snowfall'] == 0)
    if 'wind_speed' in df_clean.columns:
        conditions_list.append(df_clean['wind_speed'] < 10)
    
    if conditions_list:
        df_clean['good_biking_conditions'] = np.logical_and.reduce(conditions_list).astype(int)
    else:
        df_clean['good_biking_conditions'] = 0
    
    # 8. Time of day categories
    def time_of_day_cat(hour):
        if hour < 6:
            return 'night'
        elif hour < 12:
            return 'morning'
        elif hour < 18:
            return 'afternoon'
        else:
            return 'evening'
    
    df_clean['time_of_day'] = df_clean['hour'].apply(time_of_day_cat)
    
    # ==================================================
    # FIXED: One-hot encoding for ALL categorical variables
    # ==================================================
    
    print("   Encoding categorical variables...")
    
    # List ALL categorical columns
    all_categorical_cols = ['season', 'holiday', 'functioning_day', 'temp_category', 'time_of_day']
    categorical_cols_to_encode = [col for col in all_categorical_cols if col in df_clean.columns]
    
    print(f"   Categorical columns to encode: {categorical_cols_to_encode}")
    
    # Create dummy variables for each categorical column
    for col in categorical_cols_to_encode:
        if col in df_clean.columns:
            print(f"   Encoding {col}...")
            # Get unique values
            unique_vals = df_clean[col].unique()
            print(f"     Unique values: {unique_vals}")
            
            # Create dummy variables
            dummies = pd.get_dummies(df_clean[col], prefix=col)
            df_clean = pd.concat([df_clean, dummies], axis=1)
    
    # ==================================================
    # FIXED: Feature scaling
    # ==================================================
    
    print("   Preparing features for modeling...")
    
    # Identify numerical features (excluding target, date, and encoded categoricals)
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove non-feature columns
    non_feature_cols = ['bike_count', 'date']
    for col in non_feature_cols:
        if col in numerical_cols:
            numerical_cols.remove(col)
    
    # Remove binary (0/1) columns from scaling
    binary_cols = []
    for col in numerical_cols:
        unique_vals = df_clean[col].unique()
        if set(unique_vals).issubset({0, 1}) and len(unique_vals) <= 2:
            binary_cols.append(col)
    
    for col in binary_cols:
        numerical_cols.remove(col)
    
    print(f"   Numerical features to scale: {len(numerical_cols)}")
    print(f"   Binary features (not scaled): {len(binary_cols)}")
    
    # Scale numerical features
    if numerical_cols:
        scaler = StandardScaler()
        df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])
    
    # ==================================================
    # FIXED: Final dataset preparation
    # ==================================================
    
    # Drop original categorical columns and other unnecessary columns
    columns_to_drop = ['date', 'temp_category', 'time_of_day', 
                      'is_morning_peak', 'is_evening_peak', 'month']
    
    # Add original categorical columns to drop list
    for col in ['season', 'holiday', 'functioning_day', 'temp_category', 'time_of_day']:
        if col in df_clean.columns and col not in columns_to_drop:
            columns_to_drop.append(col)
    
    # Drop columns
    for col in columns_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
    
    # Ensure target column exists
    if 'bike_count' not in df_clean.columns:
        raise ValueError("Target column 'bike_count' not found in final dataset")
    
    # Check final dataset
    print(f"\n   Final dataset shape: {df_clean.shape}")
    print(f"   Total features: {len(df_clean.columns) - 1}")  # Excluding target
    
    # Count engineered features
    engineered_keywords = ['_sin', '_cos', '_squared', 'interaction', 
                          'is_', 'temp_', 'comfort_', 'rush_', 'good_',
                          'time_of_day_', 'season_', 'temp_category_']
    engineered_features = [col for col in df_clean.columns if any(keyword in col for keyword in engineered_keywords)]
    print(f"   Engineered features created: {len(engineered_features)}")
    
    if engineered_features:
        print(f"   First 10 engineered features: {engineered_features[:10]}")
    
    # Check for any remaining NaN values
    if df_clean.isnull().sum().sum() > 0:
        print(f"   ⚠️ Warning: Found {df_clean.isnull().sum().sum()} NaN values in final dataset")
        df_clean = df_clean.fillna(0)
    
    # Display all feature names
    print(f"\n   All features ({len(df_clean.columns)} total):")
    for i, col in enumerate(df_clean.columns.tolist()):
        print(f"     {i+1:3d}. {col}")
    
    return df_clean

def train_model_with_features():
    """Train Random Forest with advanced features"""
    
    print("\n" + "="*60)
    print("🚀 TRAINING RANDOM FOREST WITH ADVANCED FEATURE ENGINEERING")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Separate features and target
    X = df.drop(columns=['bike_count'])
    y = df['bike_count']
    
    print(f"\n📊 Dataset Information:")
    print(f"   Total samples: {len(X)}")
    print(f"   Total features: {len(X.columns)}")
    print(f"   Target range: {y.min():.0f} to {y.max():.0f} bikes")
    print(f"   Target mean: {y.mean():.1f} bikes")
    print(f"   Target std: {y.std():.1f} bikes")
    
    # Display feature information
    print(f"\n   Feature types:")
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"   Numeric features: {len(numeric_features)}")
    print(f"   Categorical features: {len(categorical_features)}")
    
    if len(categorical_features) > 0:
        print(f"   ⚠️ WARNING: Found categorical features: {categorical_features}")
        print("   Converting categorical features to numeric...")
        # Convert any remaining categorical features to numeric
        for col in categorical_features:
            X[col] = pd.factorize(X[col])[0]
    
    # Display first few feature names
    print(f"\n   First 15 features: {X.columns[:15].tolist()}")
    
    # Count engineered features
    engineered_keywords = ['_sin', '_cos', '_squared', 'interaction', 
                          'is_', 'temp_', 'comfort_', 'rush_', 'good_',
                          'time_of_day_', 'season_', 'temp_category_']
    engineered_features = [col for col in X.columns if any(keyword in col for keyword in engineered_keywords)]
    print(f"   Engineered features: {len(engineered_features)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\n📈 Train/Test Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Training features: {X_train.shape[1]}")
    
    # ==================================================
    # Train Random Forest model
    # ==================================================
    
    print("\n🤖 Training Random Forest model...")
    
    # First, test with a simple model to ensure it works
    try:
        # Test with a small subset first
        test_model = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        test_model.fit(X_train[:100], y_train[:100])
        print("   ✅ Basic model test passed!")
    except Exception as e:
        print(f"   ❌ Basic model test failed: {e}")
        # Check data types
        print(f"   X_train dtypes: {X_train.dtypes.unique()}")
        print(f"   Non-numeric columns: {X_train.select_dtypes(exclude=[np.number]).columns.tolist()}")
        raise
    
    # Now train the full model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("   Training full model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Safe MAPE calculation
    def safe_mape(y_true, y_pred):
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    train_mape = safe_mape(y_train, y_pred_train)
    test_mape = safe_mape(y_test, y_pred_test)
    
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE WITH ADVANCED FEATURES")
    print("="*60)
    print(f"\n🏋️‍♂️ Training Set:")
    print(f"   RMSE: {train_rmse:.2f} bikes")
    print(f"   R² Score: {train_r2:.4f}")
    print(f"   MAE: {train_mae:.2f} bikes")
    if not np.isnan(train_mape):
        print(f"   MAPE: {train_mape:.2f}%")
    
    print(f"\n🧪 Testing Set:")
    print(f"   RMSE: {test_rmse:.2f} bikes")
    print(f"   R² Score: {test_r2:.4f}")
    print(f"   MAE: {test_mae:.2f} bikes")
    if not np.isnan(test_mape):
        print(f"   MAPE: {test_mape:.2f}%")
    
    # Calculate improvement if we know original performance
    original_test_rmse = 112.5
    improvement = ((original_test_rmse - test_rmse) / original_test_rmse) * 100
    print(f"\n📈 Improvement vs expected original (RMSE={original_test_rmse}):")
    print(f"   RMSE reduction: {improvement:.1f}%")
    print(f"   New RMSE: {test_rmse:.1f} vs Original: {original_test_rmse:.1f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 TOP 15 MOST IMPORTANT FEATURES:")
    for i, row in feature_importance.head(15).iterrows():
        importance_percent = row['importance'] * 100
        print(f"   {i+1:2d}. {row['feature']:35s}: {importance_percent:5.1f}%")
    
    # Check if any engineered features are in top 15
    engineered_in_top = [f for f in feature_importance.head(15)['feature'].tolist() 
                        if any(keyword in f for keyword in engineered_keywords)]
    print(f"\n🔧 Engineered features in top 15: {len(engineered_in_top)}")
    if engineered_in_top:
        for feat in engineered_in_top:
            importance = feature_importance[feature_importance['feature'] == feat]['importance'].values[0] * 100
            print(f"   • {feat:35s}: {importance:5.1f}%")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/advanced_random_forest.pkl'
    joblib.dump(model, model_path)
    
    # Save feature names for later use
    feature_info = {
        'feature_names': list(X.columns),
        'feature_importance': feature_importance.to_dict('records'),
        'performance': {
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_mape': float(train_mape) if not np.isnan(train_mape) else None,
            'test_mape': float(test_mape) if not np.isnan(test_mape) else None
        },
        'dataset_info': {
            'total_samples': len(X),
            'total_features': len(X.columns),
            'engineered_features_count': len(engineered_features),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    }
    
    with open('models/advanced_model_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"📋 Model info saved to: models/advanced_model_info.json")
    
    # Compare with original model if it exists
    try:
        original_model_path = 'models/random_forest_model.pkl'
        if os.path.exists(original_model_path):
            original_model = joblib.load(original_model_path)
            print(f"\n" + "="*60)
            print("🔄 COMPARISON WITH ORIGINAL MODEL")
            print("="*60)
            
            if hasattr(original_model, 'n_features_in_'):
                print(f"   Original model features: {original_model.n_features_in_}")
                print(f"   Advanced model features: {len(X.columns)}")
                additional_features = len(X.columns) - original_model.n_features_in_
                print(f"   Additional features added: {additional_features}")
            
    except Exception as e:
        print(f"⚠️  Could not load original model for comparison: {e}")
    
    return model, X.columns, feature_importance

def analyze_feature_impact():
    """Analyze impact of new features"""
    print("\n" + "="*60)
    print("🔬 FEATURE ENGINEERING IMPACT ANALYSIS")
    print("="*60)
    
    print("\n📈 Expected Impact of Engineered Features:")
    print("   -----------------------------------------")
    print("   1. Cyclical time features (hour_sin, hour_cos):")
    print("      • Expected impact: +2-4% accuracy")
    print("      • Why: Bike demand follows daily cycles, not linear hours")
    
    print("\n   2. Temp-hour interaction:")
    print("      • Expected impact: +1-3% accuracy")
    print("      • Why: Temperature effect varies by time of day")
    
    print("\n   3. Weekend flag & Time of day:")
    print("      • Expected impact: +3-5% accuracy")
    print("      • Why: Different usage patterns on weekends and times of day")
    
    print("\n   4. Peak hour indicators:")
    print("      • Expected impact: +2-4% accuracy")
    print("      • Why: Captures commuting patterns")
    
    print("\n   5. Weather comfort features:")
    print("      • Expected impact: +2-3% accuracy")
    print("      • Why: Combined weather effects matter more than individual factors")
    
    print("\n💡 Total Expected Improvement:")
    print("   • RMSE reduction: 10-20% (from 112.5 to 90-100 bikes)")
    print("   • R² improvement: +0.05 to +0.10 (from 0.82 to 0.87-0.92)")
    print("   • Key: Interactions capture complex real-world patterns")

if __name__ == "__main__":
    try:
        # Analyze feature impact first
        analyze_feature_impact()
        
        # Train model with advanced features
        print("\n" + "="*60)
        print("🎯 STARTING ADVANCED MODEL TRAINING")
        print("="*60)
        
        model, feature_names, importance = train_model_with_features()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print("\n🎯 Next steps:")
        print("   1. Test the model: curl -X POST http://127.0.0.1:5000/predict/advanced")
        print("   2. Compare models: Visit /predict/compare endpoint")
        print("   3. Check model info: Visit /api/models")
        print("   4. Monitor RMSE improvement in logs")
        
        # Print sample prediction code
        print("\n📝 Sample test command:")
        print("""   curl -X POST http://127.0.0.1:5000/predict/advanced \\
             -H "Content-Type: application/json" \\
             -d '{
               "hour": 14,
               "hour_sin": 0.9659,
               "hour_cos": -0.2588,
               "temperature": 0.5,
               "humidity": 0.3,
               "wind_speed": 0.1,
               "visibility": 1.0,
               "dew_point": 0.2,
               "solar_radiation": 0.8,
               "rainfall": 0.0,
               "snowfall": 0.0,
               "hour_squared": 2.5,
               "temp_hour_interaction": 0.8,
               "day_of_week": 2,
               "is_weekend": 0,
               "is_peak_hour": 0,
               "comfort_index": 0.4,
               "temp_squared": 0.25,
               "wind_speed_squared": 0.01,
               "humidity_squared": 0.09,
               "rush_hour_intensity": 0.3,
               "good_biking_conditions": 1,
               "season_Summer": 1,
               "season_Winter": 0,
               "season_Spring": 0,
               "holiday_No_Holiday": 1,
               "holiday_Holiday": 0,
               "functioning_day_Yes": 1,
               "functioning_day_No": 0,
               "time_of_day_afternoon": 1,
               "time_of_day_evening": 0,
               "time_of_day_morning": 0,
               "time_of_day_night": 0
             }'""")
        
        print("\n🔍 To test with actual data from the CSV:")
        print("""   python -c "
import joblib
import pandas as pd
model = joblib.load('models/advanced_random_forest.pkl')
print(f'Model loaded: {model.n_features_in_} features')
print('First 10 feature names:', model.feature_names_in_[:10])
\"""")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("   The file 'data/raw/seoul_bike.csv' must exist in the correct location.")
        print("   Current working directory:", os.getcwd())
        print("   Expected path: data/raw/seoul_bike.csv")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Check if data/raw/seoul_bike.csv exists")
        print("   2. Check file encoding (try opening in Notepad++)")
        print("   3. Check column names in the CSV file")
        print("   4. Ensure the required packages are installed")
        print("   5. Check if the CSV file has the expected columns")
        print("\n📋 For more details, run this debug script:")
        print("""   python -c "
import pandas as pd
import os
path = 'data/raw/seoul_bike.csv'
print(f'Checking file: {path}')
print(f'File exists: {os.path.exists(path)}')
if os.path.exists(path):
    df = pd.read_csv(path, nrows=10)
    print(f'Shape: {df.shape}')
    print('Columns:', df.columns.tolist())
    print('\nData types:')
    print(df.dtypes)
    print('\nFirst 5 rows:')
    print(df.head())
else:
    print('File not found!')
\"""")
        traceback.print_exc()