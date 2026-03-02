"""
Model manager for loading and using different models
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.scaler = None
        self.feature_names = None
        
    def load_model(self, model_name='best'):
        """Load a trained model"""
        try:
            if model_name == 'best':
                # Try to load the best model from comparison
                model_path = 'models/best_model_xgboost.pkl'
            else:
                model_path = f'models/best_model_{model_name.lower().replace(" ", "_")}.pkl'
            
            print(f" Loading model from: {model_path}")
            model_package = joblib.load(model_path)
            
            self.current_model = model_package['model']
            self.scaler = model_package['scaler']
            self.model_name = model_package['model_name']
            self.test_r2 = model_package.get('test_r2', 0)
            self.test_rmse = model_package.get('test_rmse', 0)
            
            print(f" {self.model_name} loaded successfully!")
            print(f"   Test R²: {self.test_r2:.4f}")
            print(f"   Test RMSE: {self.test_rmse:.2f}")
            
            return True
            
        except Exception as e:
            print(f" Error loading model: {e}")
            # Fall back to original model
            return self.load_original_model()
    
    def load_original_model(self):
        """Load the original Random Forest model as fallback"""
        try:
            from app.utils import load_model
            self.current_model = load_model('original')
            self.model_name = 'Random Forest (Original)'
            print(f" Loaded original Random Forest model as fallback")
            return True
        except Exception as e:
            print(f" Could not load any model: {e}")
            return False
    
    def preprocess_input(self, input_data, model_type='improved'):
        """Preprocess input data for prediction"""
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Extract date features if Date is present
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
            df = df.drop('Date', axis=1)
        
        # Handle categorical variables
        categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']
        for col in categorical_cols:
            if col in df.columns:
                # Try to convert to numeric if not already
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
        
        # Ensure all expected columns are present
        expected_features = self.get_expected_features()
        for feature in expected_features:
            if feature not in df.columns:
                # Add missing feature with default value
                if 'Temperature' in feature:
                    df[feature] = 20.0  # Default temperature
                elif 'Hour' in feature:
                    df[feature] = 12    # Default hour
                else:
                    df[feature] = 0     # Default for other features
        
        # Reorder columns to match training
        df = df[expected_features]
        
        # Scale features if scaler is available
        if self.scaler:
            scaled_data = self.scaler.transform(df)
        else:
            scaled_data = df.values
        
        return scaled_data
    
    def get_expected_features(self):
        """Get the features expected by the current model"""
        if hasattr(self.current_model, 'feature_names_in_'):
            return list(self.current_model.feature_names_in_)
        else:
            # Default feature set (17 features with date features)
            return [
                'Hour', 'Temperature(캜)', 'Humidity(%)', 'Wind speed (m/s)', 
                'Visibility (10m)', 'Dew point temperature(캜)', 
                'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)',
                'Seasons', 'Holiday', 'Functioning Day',
                'Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend'
            ]
    
    def predict(self, input_data):
        """Make prediction using current model"""
        if self.current_model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.current_model.predict(processed_data)
        
        # Convert to appropriate format
        if isinstance(prediction, np.ndarray):
            predicted_bikes = float(prediction[0])
        else:
            predicted_bikes = float(prediction)
        
        return predicted_bikes
    
    def get_model_info(self):
        """Get information about the current model"""
        if self.current_model is None:
            return {"error": "No model loaded"}
        
        info = {
            "model_name": self.model_name,
            "model_type": str(type(self.current_model).__name__),
            "test_r2": getattr(self, 'test_r2', 'Unknown'),
            "test_rmse": getattr(self, 'test_rmse', 'Unknown'),
            "features_count": self.current_model.n_features_in_ if hasattr(self.current_model, 'n_features_in_') else 'Unknown',
            "is_loaded": True
        }
        
        if hasattr(self.current_model, 'feature_names_in_'):
            info["feature_names"] = list(self.current_model.feature_names_in_)
        
        return info

# Create global model manager instance
model_manager = ModelManager()