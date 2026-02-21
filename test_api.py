# test_api.py
import requests
import json

# Base URL of your Flask app
BASE_URL = "http://127.0.0.1:5000"

# Sample input data that matches what your form sends
test_data = {
    "Hour": 14,
    "Temperature(캜)": 22.5,
    "Humidity(%)": 60,
    "Wind speed (m/s)": 2.5,
    "Visibility (10m)": 2000,
    "Rainfall(mm)": 0,
    "Snowfall (cm)": 0,
    "Solar Radiation (MJ/m2)": 1.5,
    "Dew point temperature(캜)": 10,
    "Seasons_Spring": 0,
    "Seasons_Summer": 1,
    "Seasons_Autumn": 0,
    "Seasons_Winter": 0,
    "Holiday_No Holiday": 1,
    "Holiday_Holiday": 0,
    "Functioning Day_Yes": 1,
    "Functioning Day_No": 0,
    "day_of_week": 3  # Thursday
}

def test_model(model_type, calculate_features=True):
    """Test a specific model"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {model_type.upper()} model...")
    print(f"{'='*60}")
    
    # Construct URL
    url = f"{BASE_URL}/predict?model_type={model_type}"
    if calculate_features:
        url += "&calculate_features=true"
    
    print(f"📡 URL: {url}")
    print(f"📤 Data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Make request
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        
        print(f"📥 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"   Prediction: {result.get('predicted_rented_bikes', 'N/A')} bikes")
            print(f"   Confidence: {result.get('confidence', 'N/A')}%")
            print(f"   Model: {result.get('model_type', 'N/A')}")
            return True
        else:
            print(f"❌ FAILED with status {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Make sure your Flask app is running on {BASE_URL}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_model_info(model_type):
    """Check model information from debug endpoints"""
    print(f"\n{'='*60}")
    print(f"📊 Checking {model_type} model info...")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"{BASE_URL}/debug/{model_type}-feature-order")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Model loaded successfully")
                print(f"   Features: {data.get('feature_count', 'N/A')}")
                print(f"   First 5 features: {data.get('feature_order', [])[:5]}")
                return True
            else:
                print(f"❌ Error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_api_status():
    """Check API status"""
    print(f"\n{'='*60}")
    print(f"📡 Checking API Status...")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is running")
            print(f"   Models loaded: {data.get('models_loaded', False)}")
            
            # Show model availability
            models = data.get('models', {})
            for model_name, model_info in models.items():
                status = "✅" if model_info.get('available') else "❌"
                features = model_info.get('features', 'N/A')
                print(f"   {status} {model_name}: {features} features")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Make sure your Flask app is running on {BASE_URL}")
        return False

def main():
    """Main test function"""
    print("\n" + "="*70)
    print("🚲 URBANFLOW BACKEND API TESTER")
    print("="*70)
    
    # First check if the server is running
    print("\n🔍 Checking if server is running...")
    try:
        requests.get(f"{BASE_URL}/", timeout=2)
        print("✅ Server is running!")
    except:
        print("❌ Cannot connect to server. Please start your Flask app first.")
        print("   Run: python run.py")
        return
    
    # Check API status
    check_api_status()
    
    # Check model info
    check_model_info('improved')
    check_model_info('advanced')
    check_model_info('xgboost')
    
    # Test each model
    print("\n" + "="*70)
    print("🎯 TESTING PREDICTIONS")
    print("="*70)
    
    # Test original model (should work)
    test_model('original', calculate_features=False)
    
    # Test improved model (should work)
    test_model('improved', calculate_features=False)
    
    # Test advanced model
    test_model('advanced', calculate_features=True)
    
    # Test XGBoost model
    test_model('xgboost', calculate_features=True)
    
    print("\n" + "="*70)
    print("✅ Testing complete!")
    print("="*70)

if __name__ == "__main__":
    main()