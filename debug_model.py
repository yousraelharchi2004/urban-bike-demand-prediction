# debug_model.py
import pickle

print("Checking improved model pickle file...")
try:
    with open('models/improved_random_forest.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Type of loaded object: {type(model_data)}")
    print(f"Keys (if dict): {list(model_data.keys()) if isinstance(model_data, dict) else 'Not a dict'}")
    
    if isinstance(model_data, dict):
        for key, value in model_data.items():
            print(f"\nKey: {key}")
            print(f"  Type: {type(value)}")
            if hasattr(value, 'predict'):
                print(f"  Has predict method: Yes")
                if hasattr(value, 'feature_names_in_'):
                    print(f"  Feature names: {list(value.feature_names_in_)}")
            else:
                print(f"  Has predict method: No")
    
except Exception as e:
    print(f"Error: {e}")