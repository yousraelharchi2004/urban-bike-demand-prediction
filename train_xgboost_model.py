import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath='data/raw/seoul_bike.csv'):
    """Load and preprocess data using advanced feature engineering"""
    # Reuse the preprocessing function from train_with_advanced_features.py
    from train_with_advanced_features import load_and_preprocess_data
    return load_and_preprocess_data(filepath)


def train_xgboost_model():
    print("\n🚀 TRAINING XGBOOST MODEL WITH ADVANCED FEATURES")

    # Load data
    df = load_and_preprocess_data()

    # Separate features and target
    X = df.drop(columns=['bike_count'])
    y = df['bike_count']

    # Factorize any remaining categorical features
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X[col] = pd.factorize(X[col])[0]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"   Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Initialize XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n📊 XGBoost MODEL PERFORMANCE")
    print(f"   Train RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}, MAE: {train_mae:.2f}")
    print(f"   Test  RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}, MAE: {test_mae:.2f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgboost_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n💾 XGBoost model saved to: {model_path}")

    # Save feature importance and metrics
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    model_info = {
        'feature_names': list(X.columns),
        'feature_importance': feature_importance.to_dict('records'),
        'performance': {
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae)
        },
        'dataset_info': {
            'total_samples': len(X),
            'total_features': len(X.columns),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    }

    with open('models/xgboost_metrics.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("📋 XGBoost metrics and feature info saved to: models/xgboost_metrics.json")

    return model, X.columns, feature_importance


if __name__ == "__main__":
    model, features, importance = train_xgboost_model()
