# diagnostic.py
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def diagnose_model():
    print("🔍 DIAGNOSING ADVANCED MODEL PERFORMANCE")
    print("="*50)
    
    # Load the model
    model = joblib.load('models/advanced_random_forest.pkl')
    
    # Load the training data to understand scaling
    from sklearn.preprocessing import StandardScaler
    
    # Create a simple test case
    print("\n1. Testing with sample data:")
    
    # Get feature names
    feature_names = model.feature_names_in_
    print(f"   Model expects {len(feature_names)} features")
    print(f"   First 5 features: {feature_names[:5]}")
    
    # Check if all features are numeric
    print(f"\n2. Checking feature data types...")
    
    # Create a simple test input
    test_data = {}
    for feat in feature_names:
        # Set reasonable default values based on feature name
        if 'hour' in feat:
            test_data[feat] = 14
        elif 'temp' in feat:
            test_data[feat] = 22.5
        elif 'humidity' in feat:
            test_data[feat] = 60
        elif feat in ['is_weekend', 'is_peak_hour', 'good_biking_conditions']:
            test_data[feat] = 0
        elif feat in ['season_Summer', 'holiday_No Holiday', 'functioning_day_Yes']:
            test_data[feat] = 1
        elif feat in ['season_Winter', 'season_Spring', 'season_Autumn', 
                     'holiday_Holiday', 'functioning_day_No']:
            test_data[feat] = 0
        elif 'time_of_day' in feat:
            if 'afternoon' in feat:
                test_data[feat] = 1
            else:
                test_data[feat] = 0
        elif 'temp_category' in feat:
            if 'warm' in feat:
                test_data[feat] = 1
            else:
                test_data[feat] = 0
        else:
            test_data[feat] = 0
    
    # Create DataFrame
    df_test = pd.DataFrame([test_data])
    
    # Ensure all columns are in correct order
    df_test = df_test[feature_names]
    
    # Make prediction
    prediction = model.predict(df_test)
    print(f"\n3. Sample prediction: {prediction[0]:.0f} bikes")
    
    # Check feature importance
    print(f"\n4. Top 10 feature importances:")
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    
    for i, idx in enumerate(top_idx):
        print(f"   {i+1:2d}. {feature_names[idx]:30s}: {importances[idx]*100:5.1f}%")
    
    print(f"\n5. Model parameters:")
    print(f"   n_estimators: {model.n_estimators}")
    print(f"   max_depth: {model.max_depth}")
    print(f"   min_samples_split: {model.min_samples_split}")
    print(f"   min_samples_leaf: {model.min_samples_leaf}")
    
    return model, feature_names

if __name__ == "__main__":
    diagnose_model()