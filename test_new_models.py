"""
Quick test to verify new models work
"""
import sys
sys.path.append('.')  # Add project root to path

from app.model_manager import model_manager
import pandas as pd

def test_model_prediction():
    """Test model prediction with sample input"""
    print("🧪 Testing new model prediction...")
    
    # Load the best model
    success = model_manager.load_model('best')
    
    if not success:
        print("❌ Failed to load model")
        return
    
    # Sample input
    sample_input = {
        'Hour': 14,
        'Temperature(캜)': 22.5,
        'Humidity(%)': 60,
        'Wind speed (m/s)': 2.5,
        'Visibility (10m)': 2000,
        'Dew point temperature(캜)': 10,
        'Solar Radiation (MJ/m2)': 1.5,
        'Rainfall(mm)': 0,
        'Snowfall (cm)': 0,
        'Seasons': 'Summer',
        'Holiday': 'No Holiday',
        'Functioning Day': 'Yes',
        'Date': '2024-01-06'  # For date features
    }
    
    try:
        # Make prediction
        prediction = model_manager.predict(sample_input)
        
        print(f"\n✅ Model: {model_manager.model_name}")
        print(f"📊 Test R²: {model_manager.test_r2:.4f}")
        print(f"🎯 Prediction: {prediction:.1f} bikes")
        print(f"🔢 Rounded: {int(round(prediction))} bikes")
        
        # Model info
        info = model_manager.get_model_info()
        print(f"\n📋 Model Info:")
        for key, value in info.items():
            if key != 'feature_names':
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_prediction()