#!/usr/bin/env python3
"""
Train LightGBM and Stacking Ensemble models for bike demand prediction.

Uses the SAME 36-feature engineering as the existing XGBoost/Advanced models
so that `preprocess_with_advanced_features()` works at prediction time.

Outputs:
    models/lightgbm_model.pkl
    models/stacking_model.pkl
    models/lightgbm_metrics.json
    models/stacking_metrics.json
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

DATA_PATH = 'data/raw/seoul_bike.csv'
RANDOM_STATE = 42
TEST_FRACTION = 0.20


# ============================================================================
# Data loading & feature engineering (same 36 features as XGBoost model)
# ============================================================================

def load_data(filepath=DATA_PATH):
    """Load CSV with encoding fallback and standardize column names."""
    for enc in ('utf-8', 'latin-1', 'cp1252', 'ISO-8859-1'):
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"Cannot read {filepath}")

    # Standardize column names (handle Korean chars)
    rename_map = {}
    for col in df.columns:
        clean = col.strip()
        if 'Temperature' in clean and '\uce98' in clean:
            if 'Dew' in clean:
                rename_map[col] = 'Dew point temperature(C)'
            else:
                rename_map[col] = 'Temperature(C)'
        else:
            rename_map[col] = clean
    df = df.rename(columns=rename_map)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)
    print(f"Loaded {len(df)} rows  |  Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    return df


def engineer_features(df):
    """
    Produce the same 36 features that `preprocess_with_advanced_features` creates,
    applied to the full dataset for training.
    """
    out = pd.DataFrame()

    # --- Map raw column names to the 36-feature names ---
    # Find the right temperature column
    temp_col = None
    for c in df.columns:
        if 'Temperature' in c and 'Dew' not in c:
            temp_col = c
            break
    dew_col = None
    for c in df.columns:
        if 'Dew' in c:
            dew_col = c
            break

    out['hour'] = df['Hour']
    out['temperature'] = df[temp_col]
    out['Humidity%'] = df['Humidity(%)']
    out['Wind_speed_m/s'] = df['Wind speed (m/s)']
    out['visibility'] = df['Visibility (10m)']
    out['dew_point'] = df[dew_col]
    out['Solar_Radiation_MJ/m2'] = df['Solar Radiation (MJ/m2)']
    out['rainfall'] = df['Rainfall(mm)']
    out['snowfall'] = df['Snowfall (cm)']

    # --- Cyclical hour encoding ---
    out['hour_sin'] = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['hour'] / 24)
    out['hour_squared'] = out['hour'] ** 2

    # --- Day of week ---
    out['day_of_week'] = df['Date'].dt.dayofweek
    out['is_weekend'] = (out['day_of_week'] >= 5).astype(int)

    # --- Temperature / hour interactions ---
    out['temp_hour_interaction'] = out['temperature'] * out['hour']
    out['temp_squared'] = out['temperature'] ** 2

    # --- Peak hours ---
    out['is_peak_hour'] = (
        ((out['hour'] >= 7) & (out['hour'] <= 9)) |
        ((out['hour'] >= 17) & (out['hour'] <= 19))
    ).astype(int)

    out['rush_hour_intensity'] = out['hour'].map(lambda h: (
        1.0 if h in (7, 8, 17, 18)
        else 0.6 if h in (6, 9, 16, 19)
        else 0.3 if h in (10, 15, 20)
        else 0.0
    ))

    # --- Good biking conditions ---
    out['good_biking_conditions'] = (
        (out['temperature'] >= 10) & (out['temperature'] <= 30) &
        (out['rainfall'] == 0) & (out['snowfall'] == 0) &
        (out['Wind_speed_m/s'] < 10)
    ).astype(int)

    # --- Season dummies ---
    season_dummies = pd.get_dummies(df['Seasons'], prefix='season')
    for s in ['Autumn', 'Spring', 'Summer', 'Winter']:
        col_name = f'season_{s}'
        out[col_name] = season_dummies[col_name].values if col_name in season_dummies.columns else 0

    # --- Holiday dummies ---
    holiday_dummies = pd.get_dummies(df['Holiday'], prefix='holiday')
    for h in ['Holiday', 'No Holiday']:
        col_name = f'holiday_{h}'
        out[col_name] = holiday_dummies[col_name].values if col_name in holiday_dummies.columns else 0

    # --- Functioning Day dummies ---
    fd_dummies = pd.get_dummies(df['Functioning Day'], prefix='functioning_day')
    for fd in ['No', 'Yes']:
        col_name = f'functioning_day_{fd}'
        out[col_name] = fd_dummies[col_name].values if col_name in fd_dummies.columns else 0

    # --- Temperature category ---
    def temp_cat(t):
        if t < 0:
            return 'freezing'
        elif t < 10:
            return 'cold'
        elif t < 20:
            return 'cool'
        elif t < 30:
            return 'warm'
        else:
            return 'hot'

    temp_cats = out['temperature'].map(temp_cat)
    tc_dummies = pd.get_dummies(temp_cats, prefix='temp_category')
    for tc in ['cold', 'cool', 'freezing', 'hot', 'warm']:
        col_name = f'temp_category_{tc}'
        out[col_name] = tc_dummies[col_name].values if col_name in tc_dummies.columns else 0

    # --- Time of day category ---
    def time_of_day(h):
        if h < 6:
            return 'night'
        elif h < 12:
            return 'morning'
        elif h < 18:
            return 'afternoon'
        else:
            return 'evening'

    tod = out['hour'].map(time_of_day)
    tod_dummies = pd.get_dummies(tod, prefix='time_of_day')
    for tod_name in ['afternoon', 'evening', 'morning', 'night']:
        col_name = f'time_of_day_{tod_name}'
        out[col_name] = tod_dummies[col_name].values if col_name in tod_dummies.columns else 0

    # --- Target ---
    out['bike_count'] = df['Rented Bike Count'].values

    print(f"Feature engineering complete: {len(out.columns) - 1} features")
    return out


# ============================================================================
# Training
# ============================================================================

def compute_metrics(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def main():
    print("=" * 60)
    print("  TRAINING: LightGBM + Stacking Ensemble")
    print("=" * 60)

    # 1. Load and engineer features
    df_raw = load_data()
    df = engineer_features(df_raw)

    X = df.drop(columns=['bike_count'])
    y = df['bike_count'].values
    feature_names = list(X.columns)

    # 2. Chronological 80/20 split
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    os.makedirs('models', exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Train LightGBM
    # ------------------------------------------------------------------
    print("\n--- Training LightGBM ---")
    from lightgbm import LGBMRegressor

    lgbm = LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)

    lgbm_train = compute_metrics(y_train, lgbm.predict(X_train))
    lgbm_test = compute_metrics(y_test, lgbm.predict(X_test))

    print(f"  Train R2: {lgbm_train['r2']:.4f}  RMSE: {lgbm_train['rmse']:.1f}")
    print(f"  Test  R2: {lgbm_test['r2']:.4f}  RMSE: {lgbm_test['rmse']:.1f}")

    # Save LightGBM
    joblib.dump(lgbm, 'models/lightgbm_model.pkl')
    lgbm_metrics = {
        'model': 'LightGBM',
        'train': lgbm_train,
        'test': lgbm_test,
        'feature_names': feature_names,
        'n_features': len(feature_names),
    }
    with open('models/lightgbm_metrics.json', 'w') as f:
        json.dump(lgbm_metrics, f, indent=2)
    print("  Saved models/lightgbm_model.pkl")

    # ------------------------------------------------------------------
    # 4. Train Stacking Ensemble
    # ------------------------------------------------------------------
    print("\n--- Training Stacking Ensemble ---")
    print("  Base learners: RandomForest, XGBoost, LightGBM")
    print("  Meta-learner: RidgeCV")

    from xgboost import XGBRegressor

    base_rf = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt',
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    base_xgb = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    base_lgbm = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, num_leaves=31,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )

    stack = StackingRegressor(
        estimators=[
            ('rf', base_rf),
            ('xgb', base_xgb),
            ('lgbm', base_lgbm),
        ],
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1,
    )

    print("  Fitting stacking ensemble (this may take a few minutes)...")
    stack.fit(X_train, y_train)

    stack_train = compute_metrics(y_train, stack.predict(X_train))
    stack_test = compute_metrics(y_test, stack.predict(X_test))

    print(f"  Train R2: {stack_train['r2']:.4f}  RMSE: {stack_train['rmse']:.1f}")
    print(f"  Test  R2: {stack_test['r2']:.4f}  RMSE: {stack_test['rmse']:.1f}")

    # Save Stacking
    joblib.dump(stack, 'models/stacking_model.pkl')
    stack_metrics = {
        'model': 'Stacking Ensemble',
        'base_learners': ['RandomForest', 'XGBoost', 'LightGBM'],
        'meta_learner': 'RidgeCV',
        'train': stack_train,
        'test': stack_test,
        'feature_names': feature_names,
        'n_features': len(feature_names),
    }
    with open('models/stacking_metrics.json', 'w') as f:
        json.dump(stack_metrics, f, indent=2)
    print("  Saved models/stacking_model.pkl")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'Test R2':>10} {'Test RMSE':>12} {'Test MAE':>10}")
    print("  " + "-" * 57)
    print(f"  {'LightGBM':<25} {lgbm_test['r2']:>10.4f} {lgbm_test['rmse']:>12.1f} {lgbm_test['mae']:>10.1f}")
    print(f"  {'Stacking Ensemble':<25} {stack_test['r2']:>10.4f} {stack_test['rmse']:>12.1f} {stack_test['mae']:>10.1f}")
    print(f"\n  Features: {len(feature_names)} (same as XGBoost/Advanced)")
    print(f"  Split: {len(X_train)} train / {len(X_test)} test (chronological)")
    print("=" * 60)


if __name__ == '__main__':
    main()
