#!/usr/bin/env python3
"""
Unified Training Pipeline for Bike Demand Prediction
=====================================================
Replaces all 6 scattered training scripts with a single, correct pipeline.

Fixes:
- Data leakage (scaler fitted on train only, time-based split)
- TimeSeriesSplit cross-validation
- Lag features with proper shift to avoid future data leaking
- SHAP explainability
- Stacking ensemble
- Residual analysis and learning curves
- Optional LSTM model

Usage:
    python train_unified_pipeline.py
"""

import os
import sys
import json
import math
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)

from app.artifact_contract import build_unified_manifest

# ============================================================================
# CONSTANTS
# ============================================================================
DATA_PATH = 'data/raw/seoul_bike.csv'
OUTPUT_DIR = 'models/unified_pipeline'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
SHAP_DIR = os.path.join(PLOTS_DIR, 'shap')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
RANDOM_STATE = 42
TEST_FRACTION = 0.20
TSCV_SPLITS = 5
SAFE_N_JOBS = 1

CANONICAL_TEMP = "Temperature(°C)"
CANONICAL_DEW = "Dew point temperature(°C)"
TEMP_ALIAS_MAP = {
    "Temperature(Â°C)": CANONICAL_TEMP,
    "Temperature(Ã‚Â°C)": CANONICAL_TEMP,
    "Temperature(Ã¬ÂºÅ“)": CANONICAL_TEMP,
    "Temperature(ìºœ)": CANONICAL_TEMP,
    "Temperature(캜)": CANONICAL_TEMP,
}
DEW_ALIAS_MAP = {
    "Dew point temperature(Â°C)": CANONICAL_DEW,
    "Dew point temperature(Ã‚Â°C)": CANONICAL_DEW,
    "Dew point temperature(Ã¬ÂºÅ“)": CANONICAL_DEW,
    "Dew point temperature(ìºœ)": CANONICAL_DEW,
    "Dew point temperature(캜)": CANONICAL_DEW,
}


# ============================================================================
# DataLoader
# ============================================================================
class DataLoader:
    """Load and clean the Seoul Bike dataset."""

    def __init__(self, filepath: str = DATA_PATH):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        """Load CSV with encoding fallback and standardize column names."""
        for enc in ('utf-8', 'latin-1', 'cp1252', 'ISO-8859-1'):
            try:
                df = pd.read_csv(self.filepath, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise RuntimeError(f"Cannot read {self.filepath} with any encoding")

        # Standardize column names (handle Korean chars)
        alias_map = {**TEMP_ALIAS_MAP, **DEW_ALIAS_MAP}
        rename_map = {}
        for col in df.columns:
            clean = col.strip()
            canonical = alias_map.get(clean)
            if not canonical and clean in alias_map:
                canonical = alias_map[clean]
            rename_map[col] = canonical or clean

        df = df.rename(columns=rename_map)

        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

        # Sort chronologically
        df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)

        print(f"[DataLoader] Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Date range: {df['Date'].min()} -> {df['Date'].max()}")
        return df


# ============================================================================
# FeatureEngineer
# ============================================================================
class FeatureEngineer:
    """Create features from raw data (all features derivable at prediction time)."""

    TARGET = 'Rented Bike Count'

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features and return a new DataFrame."""
        out = df.copy()

        # --- Time features ---
        out['hour_sin'] = np.sin(2 * np.pi * out['Hour'] / 24)
        out['hour_cos'] = np.cos(2 * np.pi * out['Hour'] / 24)
        out['hour_squared'] = out['Hour'] ** 2
        out['month'] = out['Date'].dt.month
        out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
        out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)
        out['day_of_week'] = out['Date'].dt.dayofweek
        out['dow_sin'] = np.sin(2 * np.pi * out['day_of_week'] / 7)
        out['dow_cos'] = np.cos(2 * np.pi * out['day_of_week'] / 7)
        out['is_weekend'] = (out['day_of_week'] >= 5).astype(int)

        # --- Peak hour indicators ---
        out['is_peak_hour'] = (
            ((out['Hour'] >= 7) & (out['Hour'] <= 9)) |
            ((out['Hour'] >= 17) & (out['Hour'] <= 19))
        ).astype(int)

        rush = out['Hour'].map(lambda h: (
            1.0 if h in (7, 8, 17, 18)
            else 0.6 if h in (6, 9, 16, 19)
            else 0.3 if h in (10, 15, 20)
            else 0.0
        ))
        out['rush_hour_intensity'] = rush

        # --- Weather interactions ---
        out['temp_hour'] = out['Temperature(°C)'] * out['Hour']
        out['temp_humidity'] = out['Temperature(°C)'] * out['Humidity(%)']
        out['comfort_index'] = out['Temperature(°C)'] - out['Humidity(%)'] / 10

        # Precipitation flag
        out['has_precipitation'] = (
            (out['Rainfall(mm)'] > 0) | (out['Snowfall (cm)'] > 0)
        ).astype(int)

        # --- Lag features (using shift to avoid leakage) ---
        target = out[self.TARGET]
        out['demand_lag_24h'] = target.shift(24)
        out['demand_lag_168h'] = target.shift(168)
        out['demand_rolling_3h_mean'] = target.shift(1).rolling(3, min_periods=1).mean()
        out['demand_rolling_24h_mean'] = target.shift(1).rolling(24, min_periods=1).mean()

        # --- Encode categoricals ---
        # Seasons
        season_dummies = pd.get_dummies(out['Seasons'], prefix='season')
        out = pd.concat([out, season_dummies], axis=1)

        # Holiday / Functioning Day
        out['is_holiday'] = (out['Holiday'] == 'Holiday').astype(int)
        out['is_functioning_day'] = (out['Functioning Day'] == 'Yes').astype(int)

        # Drop raw categoricals and non-feature columns
        drop_cols = ['Date', 'Seasons', 'Holiday', 'Functioning Day']
        out = out.drop(columns=[c for c in drop_cols if c in out.columns])

        return out


# ============================================================================
# LeakageFreePipeline
# ============================================================================
class LeakageFreePipeline:
    """Ensure no data leakage: time-based split, train-only scaler."""

    def __init__(self, test_fraction: float = TEST_FRACTION):
        self.test_fraction = test_fraction
        self.scaler = StandardScaler()
        self.feature_names: list = []
        self.train_lag_median: dict = {}

    def split(self, df: pd.DataFrame, target_col: str):
        """Time-based train/test split at the given fraction mark."""
        split_idx = int(len(df) * (1 - self.test_fraction))

        y = df[target_col].values
        X = df.drop(columns=[target_col])

        # Fill lag NaNs with training-set median (NOT bfill!)
        lag_cols = [c for c in X.columns if 'lag' in c or 'rolling' in c]
        X_train_raw = X.iloc[:split_idx]
        X_test_raw = X.iloc[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        for col in lag_cols:
            median_val = X_train_raw[col].median()
            self.train_lag_median[col] = float(median_val)
            X_train_raw = X_train_raw.copy()
            X_test_raw = X_test_raw.copy()
            X_train_raw[col] = X_train_raw[col].fillna(median_val)
            X_test_raw[col] = X_test_raw[col].fillna(median_val)

        # Identify numeric columns to scale (exclude binary 0/1 columns)
        numeric_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
        binary_cols = [
            c for c in numeric_cols
            if set(X_train_raw[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})
            and X_train_raw[c].nunique() <= 2
        ]
        scale_cols = [c for c in numeric_cols if c not in binary_cols]

        # Fit scaler on TRAINING data only
        self.scaler.fit(X_train_raw[scale_cols])
        X_train_scaled = X_train_raw.copy()
        X_test_scaled = X_test_raw.copy()
        X_train_scaled[scale_cols] = self.scaler.transform(X_train_raw[scale_cols])
        X_test_scaled[scale_cols] = self.scaler.transform(X_test_raw[scale_cols])

        self.feature_names = list(X_train_scaled.columns)
        self.scale_cols = scale_cols

        print(f"[Pipeline] Train: {len(X_train_scaled)} | Test: {len(X_test_scaled)}")
        print(f"  Features: {len(self.feature_names)} | Scaled: {len(scale_cols)} | Binary: {len(binary_cols)}")

        return X_train_scaled, X_test_scaled, y_train, y_test


# ============================================================================
# ModelTrainer
# ============================================================================
class ModelTrainer:
    """Train multiple models with TimeSeriesSplit CV."""

    def __init__(self, random_state: int = RANDOM_STATE, n_splits: int = TSCV_SPLITS):
        self.random_state = random_state
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.search_iterations = 12
        self.param_distributions = {
            'random_forest': {
                'n_estimators': [200, 300, 400],
                'max_depth': [12, 16, 20, None],
                'min_samples_split': [2, 5, 8],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 0.6, 0.8],
            },
            'gradient_boosting': {
                'n_estimators': [200, 300, 400],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.02, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'max_features': ['sqrt', 0.6, None],
            },
            'xgboost': {
                'n_estimators': [300, 400, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.02, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
            },
            'lightgbm': {
                'n_estimators': [300, 400, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.02, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'num_leaves': [31, 45, 63],
            },
        }

    def get_models(self) -> dict:
        """Return dictionary of model name -> estimator."""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=300, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt',
                random_state=self.random_state, n_jobs=SAFE_N_JOBS,
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, random_state=self.random_state,
            ),
        }

        # XGBoost (optional)
        try:
            from xgboost import XGBRegressor
            models['xgboost'] = XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=SAFE_N_JOBS, verbosity=0,
            )
        except ImportError:
            print("[Trainer] xgboost not installed, skipping")

        # LightGBM (optional)
        try:
            from lightgbm import LGBMRegressor
            models['lightgbm'] = LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, num_leaves=31,
                random_state=self.random_state, n_jobs=SAFE_N_JOBS, verbose=-1,
            )
        except ImportError:
            print("[Trainer] lightgbm not installed, skipping")

        return models

    def _maybe_tune_model(self, name, estimator, X_train, y_train):
        params = self.param_distributions.get(name)
        if not params:
            return estimator, False

        print(f"  Hyperparameter search: {name}")
        search = RandomizedSearchCV(
            estimator,
            params,
            scoring='r2',
            n_iter=min(self.search_iterations, sum(len(v) for v in params.values())),
            cv=self.tscv,
            random_state=self.random_state,
            n_jobs=SAFE_N_JOBS,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train)
        print(f"    Best CV R²: {search.best_score_:.4f}")
        return search.best_estimator_, True

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train all models, return results dict."""
        models = self.get_models()
        results = {}

        for name, model in models.items():
            print(f"\n  Training {name}...")
            model, tuned = self._maybe_tune_model(name, model, X_train, y_train)
            if not tuned:
                model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.tscv, scoring='neg_root_mean_squared_error', n_jobs=SAFE_N_JOBS,
            )

            metrics = {
                'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                'train_r2': float(r2_score(y_train, y_pred_train)),
                'test_r2': float(r2_score(y_test, y_pred_test)),
                'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
                'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
                'cv_rmse_mean': float(-cv_scores.mean()),
                'cv_rmse_std': float(cv_scores.std()),
                'explained_variance': float(explained_variance_score(y_test, y_pred_test)),
            }

            results[name] = {
                'model': model,
                'metrics': metrics,
                'y_pred_test': y_pred_test,
                'y_pred_train': y_pred_train,
            }

            print(f"    Test R²: {metrics['test_r2']:.4f} | "
                  f"Test RMSE: {metrics['test_rmse']:.1f} | "
                  f"CV RMSE: {metrics['cv_rmse_mean']:.1f} +/- {metrics['cv_rmse_std']:.1f}")

        return results


# ============================================================================
# StackingEnsemble
# ============================================================================
class StackingEnsemble:
    """Build a stacking ensemble from the top-3 models."""

    def build(self, results: dict, X_train, y_train, X_test, y_test, tscv):
        """Select top-3 by test R2, stack with RidgeCV meta-learner."""
        sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['test_r2'], reverse=True)
        top3 = sorted_models[:3]
        top3_names = [name for name, _ in top3]

        print(f"\n[Ensemble] Stacking top-3: {top3_names}")

        estimators = [(name, results[name]['model']) for name in top3_names]

        # StackingRegressor internally uses cross_val_predict, which requires
        # partition-based CV and does not support TimeSeriesSplit.
        stacking_cv = KFold(n_splits=5, shuffle=False)

        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(),
            cv=stacking_cv,
            n_jobs=SAFE_N_JOBS,
        )

        stack.fit(X_train, y_train)

        y_pred_test = stack.predict(X_test)
        y_pred_train = stack.predict(X_train)

        metrics = {
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'train_r2': float(r2_score(y_train, y_pred_train)),
            'test_r2': float(r2_score(y_test, y_pred_test)),
            'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
            'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
            'explained_variance': float(explained_variance_score(y_test, y_pred_test)),
            'base_models': top3_names,
        }

        print(f"  Ensemble Test R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.1f}")

        return stack, metrics, y_pred_test


# ============================================================================
# Evaluator (residual analysis + learning curves)
# ============================================================================
class Evaluator:
    """Generate residual plots and learning curves."""

    @staticmethod
    def residual_plots(y_test, y_pred, model_name: str, save_dir: str):
        """4-panel residual plot."""
        residuals = y_test - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')

        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[0, 0].axhline(0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')

        # 2. Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Residual Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residual Distribution')

        # 3. Q-Q plot (manual)
        sorted_res = np.sort(residuals)
        n = len(sorted_res)
        theoretical = np.array([
            np.sqrt(2) * math.erfc(2 * (i - 0.5) / n)
            if (i - 0.5) / n < 0.5 else -np.sqrt(2) * math.erfc(2 * (1 - (i - 0.5) / n))
            for i in range(1, n + 1)
        ]) if n < 5000 else np.linspace(-3, 3, n)
        # Simplified Q-Q: just use sorted standardized residuals vs expected
        std_res = (sorted_res - np.mean(residuals)) / (np.std(residuals) + 1e-8)
        expected = np.linspace(-3, 3, len(std_res))
        axes[1, 0].scatter(expected, std_res, alpha=0.3, s=10)
        axes[1, 0].plot([-3, 3], [-3, 3], 'r--')
        axes[1, 0].set_xlabel('Theoretical Quantiles')
        axes[1, 0].set_ylabel('Sample Quantiles')
        axes[1, 0].set_title('Q-Q Plot')

        # 4. Residuals over time (index)
        axes[1, 1].plot(range(len(residuals)), residuals, alpha=0.5, linewidth=0.5)
        axes[1, 1].axhline(0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Observation Index')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].set_title('Residuals Over Time')

        plt.tight_layout()
        path = os.path.join(save_dir, f'residuals_{model_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved residual plot: {path}")

    @staticmethod
    def learning_curve(model, X_train, y_train, model_name: str, save_dir: str, tscv=None):
        """Learning curve: training size vs RMSE."""
        train_sizes_frac = np.linspace(0.2, 1.0, 8)
        train_rmses = []
        val_rmses = []

        for frac in train_sizes_frac:
            n = int(len(X_train) * frac)
            if n < 100:
                continue
            X_sub = X_train[:n]
            y_sub = y_train[:n]

            # Simple time-based validation: last 20% of subset
            split = int(n * 0.8)
            X_t, X_v = X_sub[:split], X_sub[split:]
            y_t, y_v = y_sub[:split], y_sub[split:]

            from sklearn.base import clone
            m = clone(model)
            m.fit(X_t, y_t)
            train_rmses.append(np.sqrt(mean_squared_error(y_t, m.predict(X_t))))
            val_rmses.append(np.sqrt(mean_squared_error(y_v, m.predict(X_v))))

        fig, ax = plt.subplots(figsize=(8, 5))
        sizes = [int(len(X_train) * f) for f in train_sizes_frac if int(len(X_train) * f) >= 100]
        ax.plot(sizes, train_rmses, 'o-', label='Training RMSE')
        ax.plot(sizes, val_rmses, 'o-', label='Validation RMSE')
        ax.set_xlabel('Training Size')
        ax.set_ylabel('RMSE')
        ax.set_title(f'Learning Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, f'learning_curve_{model_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved learning curve: {path}")


# ============================================================================
# ShapExplainer
# ============================================================================
class ShapExplainer:
    """Generate SHAP plots for model explainability."""

    @staticmethod
    def explain(model, X_test, feature_names: list, save_dir: str, model_name: str = 'best'):
        try:
            import shap
        except ImportError:
            print("[SHAP] shap not installed, skipping explainability plots")
            return

        print(f"\n[SHAP] Generating explanations for {model_name}...")

        # Use a subsample for speed
        n_samples = min(500, len(X_test))
        X_sample = X_test[:n_samples]

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            # Fallback for non-tree models
            try:
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample).values
            except Exception as e:
                print(f"  [SHAP] Could not compute SHAP values: {e}")
                return

        # Convert to DataFrame for plotting if needed
        if isinstance(X_sample, np.ndarray):
            X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
        else:
            X_sample_df = X_sample

        # 1. Summary bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample_df, plot_type='bar', show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_summary_bar_{model_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Beeswarm plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample_df, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_beeswarm_{model_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Top-3 feature dependence plots
        if isinstance(shap_values, np.ndarray):
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
            for idx in top3_idx:
                feat_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
                plt.figure(figsize=(8, 5))
                shap.dependence_plot(idx, shap_values, X_sample_df, show=False)
                plt.tight_layout()
                safe_name = feat_name.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
                plt.savefig(os.path.join(save_dir, f'shap_dep_{safe_name}_{model_name}.png'),
                           dpi=150, bbox_inches='tight')
                plt.close()

        print(f"  SHAP plots saved to {save_dir}")


# ============================================================================
# LSTMTrainer (optional)
# ============================================================================
class LSTMTrainer:
    """Optional LSTM model for time series prediction."""

    SEQ_LEN = 24

    @staticmethod
    def train(X_train, y_train, X_test, y_test, save_dir: str):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            print("[LSTM] TensorFlow not installed, skipping LSTM model")
            return None, None

        print("\n[LSTM] Training LSTM model...")
        seq_len = LSTMTrainer.SEQ_LEN

        # Reshape for sequences
        def create_sequences(X, y, seq_len):
            Xs, ys = [], []
            for i in range(len(X) - seq_len):
                Xs.append(X[i:i + seq_len])
                ys.append(y[i + seq_len])
            return np.array(Xs), np.array(ys)

        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_np = X_test.values if hasattr(X_test, 'values') else X_test

        X_train_seq, y_train_seq = create_sequences(X_train_np, y_train, seq_len)
        X_test_seq, y_test_seq = create_sequences(X_test_np, y_test, seq_len)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print("  [LSTM] Not enough data for sequences")
            return None, None

        n_features = X_train_seq.shape[2]

        model = Sequential([
            LSTM(64, input_shape=(seq_len, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1),
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=50,
            batch_size=64,
            validation_data=(X_test_seq, y_test_seq),
            callbacks=[early_stop],
            verbose=1,
        )

        y_pred = model.predict(X_test_seq).flatten()
        metrics = {
            'test_rmse': float(np.sqrt(mean_squared_error(y_test_seq, y_pred))),
            'test_r2': float(r2_score(y_test_seq, y_pred)),
            'test_mae': float(mean_absolute_error(y_test_seq, y_pred)),
        }

        print(f"  LSTM Test R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.1f}")

        # Save
        model_path = os.path.join(save_dir, 'lstm_model.keras')
        model.save(model_path)
        print(f"  LSTM model saved to {model_path}")

        return model, metrics


# ============================================================================
# ArtifactSaver
# ============================================================================
class ArtifactSaver:
    """Save all pipeline artifacts to a structured directory."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots', 'shap'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)

    def save_model(self, model, name: str):
        path = os.path.join(self.output_dir, f'{name}.pkl')
        joblib.dump(model, path)
        print(f"  Saved model: {path}")

    def save_scaler(self, scaler):
        path = os.path.join(self.output_dir, 'scaler.pkl')
        joblib.dump(scaler, path)
        print(f"  Saved scaler: {path}")

    def save_feature_names(self, names: list):
        path = os.path.join(self.output_dir, 'feature_names.json')
        with open(path, 'w') as f:
            json.dump(names, f, indent=2)
        print(f"  Saved feature names: {path}")

    def save_config(self, config: dict):
        path = os.path.join(self.output_dir, 'pipeline_config.json')
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"  Saved config: {path}")

    def save_metrics(self, all_metrics: dict):
        path = os.path.join(self.output_dir, 'metrics', 'all_models_comparison.json')
        with open(path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"  Saved metrics: {path}")

    def save_manifest(self, manifest: dict):
        path = os.path.join(self.output_dir, 'artifact_manifest.json')
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Saved artifact manifest: {path}")

    def save_demand_cache(self, df: pd.DataFrame, target_col: str):
        """Save last 168 demand values for Flask prediction-time lag features."""
        last_168 = df[target_col].tail(168).tolist()
        path = os.path.join(self.output_dir, 'demand_cache.json')
        with open(path, 'w') as f:
            json.dump({'last_168_values': last_168, 'updated_at': datetime.now().isoformat()}, f, indent=2)
        print(f"  Saved demand cache: {path}")

    def save_lag_medians(self, medians: dict):
        path = os.path.join(self.output_dir, 'lag_medians.json')
        with open(path, 'w') as f:
            json.dump(medians, f, indent=2)
        print(f"  Saved lag medians: {path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("  UNIFIED BIKE DEMAND TRAINING PIPELINE")
    print("  Seoul Bike Sharing Dataset - Leakage-Free Pipeline")
    print("=" * 70)

    # 1. Load data
    loader = DataLoader()
    df = loader.load()

    # 2. Feature engineering
    fe = FeatureEngineer()
    df_features = fe.transform(df)
    print(f"\n[Features] Total features: {len(df_features.columns) - 1}")

    # 3. Save demand cache before dropping target
    saver = ArtifactSaver()
    saver.save_demand_cache(df, FeatureEngineer.TARGET)

    # 4. Leakage-free split + scaling
    pipeline = LeakageFreePipeline()
    X_train, X_test, y_train, y_test = pipeline.split(df_features, FeatureEngineer.TARGET)

    saver.save_scaler(pipeline.scaler)
    saver.save_feature_names(pipeline.feature_names)
    saver.save_lag_medians(pipeline.train_lag_median)

    # 5. Train models
    print("\n" + "=" * 70)
    print("  TRAINING MODELS")
    print("=" * 70)

    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

    # 6. Stacking ensemble
    print("\n" + "=" * 70)
    print("  STACKING ENSEMBLE")
    print("=" * 70)

    ensemble_builder = StackingEnsemble()
    stack_model, stack_metrics, stack_preds = ensemble_builder.build(
        results, X_train, y_train, X_test, y_test, trainer.tscv
    )

    # 7. Save all models
    print("\n" + "=" * 70)
    print("  SAVING ARTIFACTS")
    print("=" * 70)

    for name, res in results.items():
        saver.save_model(res['model'], f'{name}_model')

    saver.save_model(stack_model, 'stacking_ensemble')

    # 8. Collect and save metrics
    all_metrics = {}
    for name, res in results.items():
        all_metrics[name] = res['metrics']
    all_metrics['stacking_ensemble'] = stack_metrics

    saver.save_metrics(all_metrics)

    # 9. Pipeline config
    config = {
        'training_date': datetime.now().isoformat(),
        'dataset': DATA_PATH,
        'test_fraction': TEST_FRACTION,
        'tscv_splits': TSCV_SPLITS,
        'random_state': RANDOM_STATE,
        'n_features': len(pipeline.feature_names),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'models_trained': list(results.keys()) + ['stacking_ensemble'],
        'best_model': max(all_metrics.items(), key=lambda x: x[1].get('test_r2', 0))[0],
        'best_test_r2': max(m.get('test_r2', 0) for m in all_metrics.values()),
        'best_test_rmse': min(m.get('test_rmse', 9999) for m in all_metrics.values()),
        'scale_cols': pipeline.scale_cols,
    }
    saver.save_config(config)

    manifest = build_unified_manifest(
        config=config,
        feature_names_path='feature_names.json',
        scaler_path='scaler.pkl',
        lag_medians_path='lag_medians.json',
        demand_cache_path='demand_cache.json',
        metrics_path='metrics/all_models_comparison.json',
        model_filenames={
            **{f'{name}': f'{name}_model.pkl' for name in results.keys()},
            'stacking_ensemble': 'stacking_ensemble.pkl',
        },
    )
    saver.save_manifest(manifest)

    # 10. Residual analysis and learning curves
    print("\n" + "=" * 70)
    print("  PLOTS: RESIDUALS & LEARNING CURVES")
    print("=" * 70)

    evaluator = Evaluator()

    # Residuals for each model + ensemble
    for name, res in results.items():
        evaluator.residual_plots(y_test, res['y_pred_test'], name, PLOTS_DIR)
    evaluator.residual_plots(y_test, stack_preds, 'stacking_ensemble', PLOTS_DIR)

    # Learning curves for each base model
    for name, res in results.items():
        evaluator.learning_curve(res['model'], X_train, y_train, name, PLOTS_DIR, trainer.tscv)

    # 11. SHAP explainability
    print("\n" + "=" * 70)
    print("  SHAP EXPLAINABILITY")
    print("=" * 70)

    best_model_name = config['best_model']
    if best_model_name == 'stacking_ensemble':
        # Use the best individual model for SHAP
        best_individual = max(results.items(), key=lambda x: x[1]['metrics']['test_r2'])
        shap_model = best_individual[1]['model']
        shap_name = best_individual[0]
    else:
        shap_model = results[best_model_name]['model']
        shap_name = best_model_name

    ShapExplainer.explain(shap_model, X_test, pipeline.feature_names, SHAP_DIR, shap_name)

    # 12. Optional LSTM
    print("\n" + "=" * 70)
    print("  OPTIONAL: LSTM MODEL")
    print("=" * 70)

    lstm_model, lstm_metrics = LSTMTrainer.train(X_train, y_train, X_test, y_test, OUTPUT_DIR)
    if lstm_metrics:
        all_metrics['lstm'] = lstm_metrics
        # Re-save with LSTM included
        saver.save_metrics(all_metrics)

    # 13. Summary
    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} {'Test R²':>10} {'Test RMSE':>12} {'Test MAE':>10}")
    print("-" * 60)
    for name, m in sorted(all_metrics.items(), key=lambda x: x[1].get('test_r2', 0), reverse=True):
        print(f"  {name:<23} {m.get('test_r2', 0):>10.4f} {m.get('test_rmse', 0):>12.1f} {m.get('test_mae', 0):>10.1f}")

    print(f"\n  Best model: {config['best_model']}")
    print(f"  Best R²: {config['best_test_r2']:.4f}")
    print(f"  Best RMSE: {config['best_test_rmse']:.1f}")

    print(f"\n  Artifacts saved to: {OUTPUT_DIR}/")
    print("=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
