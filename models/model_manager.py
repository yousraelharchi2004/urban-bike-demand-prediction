import joblib

# Load models once (efficient)
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")

def predict_rf(features):
    return rf_model.predict([features])[0]

def predict_xgb(features):
    return xgb_model.predict([features])[0]
