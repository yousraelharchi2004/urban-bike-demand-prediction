"""
Artifact-backed model evaluation metrics for the Bike Demand Prediction System.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path

import joblib
import numpy as np

from app.artifact_contract import read_manifest, unified_manifest_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / 'models'
UNIFIED_ARTIFACT_DIR = MODELS_DIR / 'unified_pipeline'
UNIFIED_METRICS_FILE = UNIFIED_ARTIFACT_DIR / 'metrics' / 'all_models_comparison.json'
UNIFIED_CONFIG_FILE = UNIFIED_ARTIFACT_DIR / 'pipeline_config.json'
UNIFIED_FEATURES_FILE = UNIFIED_ARTIFACT_DIR / 'feature_names.json'
UNIFIED_MODEL_ALIAS = {
    'original': 'random_forest',
    'improved': 'gradient_boosting',
    'advanced': 'xgboost',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'stacking': 'stacking_ensemble',
}

MODEL_DISPLAY = {
    'original': {
        'name': 'Random Forest (baseline)',
        'description': 'Original baseline random forest model',
        'feature_engineering': 'None',
        'recommended_for': 'Quick baseline predictions',
        'speed': 95,
    },
    'improved': {
        'name': 'Improved RF (engineered)',
        'description': 'Random forest with engineered time and interaction features',
        'feature_engineering': 'Basic engineered features',
        'recommended_for': 'Higher accuracy than baseline',
        'speed': 90,
    },
    'advanced': {
        'name': 'Advanced RF',
        'description': 'Advanced random forest with expanded engineered feature space',
        'feature_engineering': 'Advanced engineered features',
        'recommended_for': 'Complex nonlinear demand patterns',
        'speed': 85,
    },
    'xgboost': {
        'name': 'XGBoost',
        'description': 'Gradient boosting model using the advanced feature set',
        'feature_engineering': 'Advanced engineered features',
        'recommended_for': 'Best tree-boosted model path',
        'speed': 80,
    },
    'lightgbm': {
        'name': 'LightGBM',
        'description': 'LightGBM regressor with advanced features',
        'feature_engineering': 'Advanced engineered features',
        'recommended_for': 'Fast boosted inference',
        'speed': 92,
    },
    'stacking': {
        'name': 'Stacking Ensemble',
        'description': 'Stacked ensemble over the advanced model family',
        'feature_engineering': 'Advanced engineered features',
        'recommended_for': 'Ensemble scoring',
        'speed': 70,
    },
    'unified': {
        'name': 'Unified Pipeline Ensemble',
        'description': 'Leakage-safe unified pipeline artifact bundle',
        'feature_engineering': 'Pipeline-managed',
        'recommended_for': 'Preferred deployable pipeline',
        'speed': 82,
    },
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _safe_number(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _load_unified_metrics() -> dict:
    return _read_json(UNIFIED_METRICS_FILE)


def _load_unified_config() -> dict:
    return _read_json(UNIFIED_CONFIG_FILE)


def _load_unified_features() -> list:
    data = _read_json(UNIFIED_FEATURES_FILE)
    return data if isinstance(data, list) else []


def _build_unified_record(alias_key: str, payload: dict, config: dict, model_key: str) -> dict:
    feature_names = _load_unified_features()
    record = {
        'r2': _safe_number(payload.get('test_r2')),
        'rmse': _safe_number(payload.get('test_rmse')),
        'mae': _safe_number(payload.get('test_mae')),
        'train_r2': _safe_number(payload.get('train_r2')),
        'train_rmse': _safe_number(payload.get('train_rmse')),
        'train_mae': _safe_number(payload.get('train_mae')),
        'features': len(feature_names),
        'training_samples': _safe_int(config.get('n_train_samples')),
        'testing_samples': _safe_int(config.get('n_test_samples')),
        'artifact_path': str(UNIFIED_METRICS_FILE),
        'alias': alias_key,
        'dataset': config.get('dataset'),
    }
    if model_key == 'stacking':
        record['stacking_members'] = payload.get('base_models', [])
    return record


def _unified_metrics_for(model_key: str) -> dict | None:
    metrics = _load_unified_metrics()
    if not metrics:
        return None
    alias = UNIFIED_MODEL_ALIAS.get(model_key)
    if not alias:
        return None
    payload = metrics.get(alias)
    if not payload:
        return None
    config = _load_unified_config()
    return _build_unified_record(alias, payload, config, model_key)


def _unified_summary() -> dict | None:
    metrics = _load_unified_metrics()
    if not metrics:
        return None
    config = _load_unified_config()
    best_alias = max(metrics.items(), key=lambda item: item[1].get('test_r2', -1))[0]
    payload = metrics.get(best_alias)
    if not payload:
        return None
    record = _build_unified_record(best_alias, payload, config, 'unified')
    record['best_model'] = best_alias
    record['artifact_path'] = str(UNIFIED_CONFIG_FILE if UNIFIED_CONFIG_FILE.exists() else UNIFIED_METRICS_FILE)
    return record


def _model_artifact_metrics(model_key: str) -> dict:
    if model_key == 'unified':
        unified_summary = _unified_summary()
        if unified_summary:
            return unified_summary

    unified_metrics = _unified_metrics_for(model_key)
    if unified_metrics:
        return unified_metrics

    if model_key == 'original':
        payload = _read_json(MODELS_DIR / 'proper_model_metrics.json')
        return {
            'r2': _safe_number(payload.get('r2')),
            'rmse': _safe_number(payload.get('rmse')),
            'mae': _safe_number(payload.get('mae')),
            'mape': _safe_number(payload.get('mape')),
            'features': _safe_int(payload.get('n_features'), 14),
            'training_samples': _safe_int(payload.get('n_train_samples'), 7008),
            'testing_samples': _safe_int(payload.get('n_test_samples'), 1752),
            'artifact_path': str(MODELS_DIR / 'proper_model_metrics.json'),
        }

    if model_key == 'improved':
        payload = _read_json(MODELS_DIR / 'improved_model_metrics.json')
        return {
            'r2': _safe_number(payload.get('r2')),
            'rmse': _safe_number(payload.get('rmse')),
            'mae': _safe_number(payload.get('mae')),
            'mape': _safe_number(payload.get('mape')),
            'features': _safe_int(payload.get('n_features'), 38),
            'artifact_path': str(MODELS_DIR / 'improved_model_metrics.json'),
        }

    if model_key == 'advanced':
        payload = _read_json(MODELS_DIR / 'advanced_model_info.json')
        perf = payload.get('performance', {})
        dataset = payload.get('dataset_info', {})
        return {
            'r2': _safe_number(perf.get('test_r2')),
            'rmse': _safe_number(perf.get('test_rmse')),
            'mae': _safe_number(perf.get('test_mae')),
            'mape': _safe_number(perf.get('test_mape')),
            'train_r2': _safe_number(perf.get('train_r2')),
            'train_rmse': _safe_number(perf.get('train_rmse')),
            'train_mae': _safe_number(perf.get('train_mae')),
            'features': _safe_int(dataset.get('total_features') or len(payload.get('feature_names', [])), 36),
            'training_samples': _safe_int(dataset.get('train_samples'), 7008),
            'testing_samples': _safe_int(dataset.get('test_samples'), 1752),
            'artifact_path': str(MODELS_DIR / 'advanced_model_info.json'),
        }

    if model_key == 'xgboost':
        payload = _read_json(MODELS_DIR / 'xgboost_metrics.json')
        perf = payload.get('performance', {})
        dataset = payload.get('dataset_info', {})
        return {
            'r2': _safe_number(perf.get('test_r2')),
            'rmse': _safe_number(perf.get('test_rmse')),
            'mae': _safe_number(perf.get('test_mae')),
            'train_r2': _safe_number(perf.get('train_r2')),
            'train_rmse': _safe_number(perf.get('train_rmse')),
            'train_mae': _safe_number(perf.get('train_mae')),
            'features': _safe_int(dataset.get('total_features') or len(payload.get('feature_names', [])), 36),
            'training_samples': _safe_int(dataset.get('train_samples'), 7008),
            'testing_samples': _safe_int(dataset.get('test_samples'), 1752),
            'artifact_path': str(MODELS_DIR / 'xgboost_metrics.json'),
        }

    if model_key == 'lightgbm':
        payload = _read_json(MODELS_DIR / 'lightgbm_metrics.json')
        train = payload.get('train', {})
        test = payload.get('test', {})
        return {
            'r2': _safe_number(test.get('r2')),
            'rmse': _safe_number(test.get('rmse')),
            'mae': _safe_number(test.get('mae')),
            'train_r2': _safe_number(train.get('r2')),
            'train_rmse': _safe_number(train.get('rmse')),
            'train_mae': _safe_number(train.get('mae')),
            'features': _safe_int(payload.get('n_features'), 36),
            'artifact_path': str(MODELS_DIR / 'lightgbm_metrics.json'),
        }

    if model_key == 'stacking':
        payload = _read_json(MODELS_DIR / 'stacking_metrics.json')
        train = payload.get('train', {})
        test = payload.get('test', {})
        return {
            'r2': _safe_number(test.get('r2')),
            'rmse': _safe_number(test.get('rmse')),
            'mae': _safe_number(test.get('mae')),
            'train_r2': _safe_number(train.get('r2')),
            'train_rmse': _safe_number(train.get('rmse')),
            'train_mae': _safe_number(train.get('mae')),
            'features': _safe_int(payload.get('n_features'), 36),
            'artifact_path': str(MODELS_DIR / 'stacking_metrics.json'),
        }

    if model_key == 'unified':
        manifest = read_manifest(unified_manifest_path(PROJECT_ROOT))
        base_dir = unified_manifest_path(PROJECT_ROOT).parent
        config = _read_json(base_dir / manifest.get('config_path', 'pipeline_config.json'))
        comparison = _read_json(base_dir / manifest.get('metrics_path', 'metrics/all_models_comparison.json'))
        best_model = config.get('best_model', 'stacking_ensemble')
        best_metrics = comparison.get(best_model, {})
        return {
            'name_suffix': best_model,
            'r2': _safe_number(config.get('best_test_r2', best_metrics.get('test_r2'))),
            'rmse': _safe_number(config.get('best_test_rmse', best_metrics.get('test_rmse'))),
            'mae': _safe_number(best_metrics.get('test_mae')),
            'train_r2': _safe_number(best_metrics.get('train_r2')),
            'train_rmse': _safe_number(best_metrics.get('train_rmse')),
            'train_mae': _safe_number(best_metrics.get('train_mae')),
            'features': _safe_int(config.get('n_features'), 35),
            'training_samples': _safe_int(config.get('n_train_samples'), 7008),
            'testing_samples': _safe_int(config.get('n_test_samples'), 1752),
            'artifact_path': str(base_dir / manifest.get('config_path', 'pipeline_config.json')),
            'best_model': best_model,
        }

    return {}


def _stability_score(metrics: dict) -> int:
    train_r2 = metrics.get('train_r2')
    test_r2 = metrics.get('r2')
    if train_r2 is None or test_r2 is None:
        return 80
    gap = abs(float(train_r2) - float(test_r2))
    return max(0, min(100, round((1 - gap) * 100)))


def get_model_catalog() -> dict[str, dict]:
    """Return artifact-backed metrics and display metadata for all models."""
    catalog: dict[str, dict] = {}
    for model_key, display in MODEL_DISPLAY.items():
        artifact_metrics = _model_artifact_metrics(model_key)
        r2 = artifact_metrics.get('r2', 0.0)
        catalog[model_key] = {
            'key': model_key,
            'name': display['name'],
            'description': display['description'],
            'feature_engineering': display['feature_engineering'],
            'recommended_for': display['recommended_for'],
            'r2': r2,
            'rmse': artifact_metrics.get('rmse', 0.0),
            'mae': artifact_metrics.get('mae', 0.0),
            'mape': artifact_metrics.get('mape'),
            'features': artifact_metrics.get('features', 0),
            'accuracy': round(r2 * 100, 1),
            'stability': _stability_score(artifact_metrics),
            'speed': display['speed'],
            'train_r2': artifact_metrics.get('train_r2'),
            'train_rmse': artifact_metrics.get('train_rmse'),
            'train_mae': artifact_metrics.get('train_mae'),
            'training_samples': artifact_metrics.get('training_samples'),
            'testing_samples': artifact_metrics.get('testing_samples'),
            'artifact_path': artifact_metrics.get('artifact_path'),
            'best_model': artifact_metrics.get('best_model'),
            'test_r2': r2,
            'test_rmse': artifact_metrics.get('rmse', 0.0),
            'test_mae': artifact_metrics.get('mae', 0.0),
            'model_available': bool(artifact_metrics),
        }
    return catalog


def get_model_metrics(model_key: str = 'original') -> dict:
    """Return normalized metrics for a single model."""
    return get_model_catalog().get(model_key, {})


def calculate_model_metrics():
    """Return artifact-backed metrics plus loaded-model metadata for the best available model."""
    catalog = get_model_catalog()
    best_model_key = max(catalog, key=lambda key: catalog[key].get('r2', -1)) if catalog else 'original'
    selected = catalog.get(best_model_key, catalog.get('original', {}))
    model_path = MODELS_DIR / 'random_forest_model.pkl'
    proper_model_path = MODELS_DIR / 'proper_random_forest.pkl'
    resolved_model_path = proper_model_path if proper_model_path.exists() else model_path

    model = None
    load_error = None
    model_available = resolved_model_path.exists()
    if model_available:
        try:
            model = joblib.load(resolved_model_path)
        except Exception as exc:
            load_error = str(exc)
            model_available = False

    metrics = {
        "model_name": selected.get('name', 'Random Forest (baseline)'),
        "model_type": type(model).__name__ if model is not None else "RandomForestRegressor",
        "model_available": model_available,
        "model_status": "Loaded successfully" if model_available else "Artifact metrics available; model load failed" if load_error else "Model file not found on disk",
        "model_path": str(resolved_model_path),
        "load_error": load_error,
        "features_count": selected.get('features', 0),
        "feature_names": list(model.feature_names_in_) if model is not None and hasattr(model, 'feature_names_in_') else None,
        "n_estimators": model.n_estimators if model is not None and hasattr(model, 'n_estimators') else None,
        "max_depth": model.max_depth if model is not None and hasattr(model, 'max_depth') else None,
        "min_samples_split": model.min_samples_split if model is not None and hasattr(model, 'min_samples_split') else None,
        "min_samples_leaf": model.min_samples_leaf if model is not None and hasattr(model, 'min_samples_leaf') else None,
        "random_state": model.random_state if model is not None and hasattr(model, 'random_state') else 42,
        "dataset": "Seoul Bike Sharing Dataset",
        "dataset_size": "8,760 records",
        "data_period": "2017-2018",
        "data_location": "Seoul, South Korea",
        "r2_score": selected.get('r2', 0.0),
        "mean_absolute_error": selected.get('mae', 0.0),
        "root_mean_squared_error": selected.get('rmse', 0.0),
        "mean_absolute_percentage_error": f"{selected.get('mape', 0.0):.1f}%" if selected.get('mape') is not None else "N/A",
        "explained_variance_score": selected.get('r2', 0.0),
        "training_samples": selected.get('training_samples', 7008),
        "testing_samples": selected.get('testing_samples', 1752),
        "validation_split": "80/20",
        "has_feature_importance": bool(model is not None and hasattr(model, 'feature_importances_')),
        "artifact_path": selected.get('artifact_path'),
        "models": catalog,
        "original_model_metrics": catalog.get('original', {}),
        "improved_model_metrics": catalog.get('improved', {}),
        "advanced_model_metrics": catalog.get('advanced', {}),
        "xgboost_model_metrics": catalog.get('xgboost', {}),
        "lightgbm_model_metrics": catalog.get('lightgbm', {}),
        "stacking_model_metrics": catalog.get('stacking', {}),
        "unified_model_metrics": catalog.get('unified', {}),
        "best_model": best_model_key,
        "interpretation": {
            "r2_interpretation": f"The best available model explains {round(selected.get('r2', 0.0) * 100, 1)}% of the variance in bike rental demand",
            "mae_interpretation": f"On average, predictions are off by {round(selected.get('mae', 0.0), 1)} bikes",
            "rmse_interpretation": f"The typical prediction error is about {round(selected.get('rmse', 0.0), 1)} bikes",
        }
    }

    if model_available and hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        metrics["top_features"] = [{
            "rank": rank,
            "feature": features[idx],
            "importance": float(importances[idx]),
            "importance_percent": f"{importances[idx] * 100:.1f}%"
        } for rank, idx in enumerate(indices, start=1)]
        metrics["feature_importance_stats"] = {
            "max_importance": float(np.max(importances)),
            "min_importance": float(np.min(importances)),
            "mean_importance": float(np.mean(importances)),
            "std_importance": float(np.std(importances))
        }

    return metrics


def get_top_features(model=None, top_n=5):
    """Extract top N most important features from the model."""
    try:
        if model is None:
            model_path = MODELS_DIR / 'random_forest_model.pkl'
            if not model_path.exists():
                return None
            model = joblib.load(model_path)

        if model and hasattr(model, 'feature_names_in_') and hasattr(model, 'feature_importances_'):
            features = model.feature_names_in_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            return [{
                "feature": features[i],
                "importance": float(importances[i]),
                "importance_percent": f"{importances[i] * 100:.1f}%"
            } for i in indices]
        return None
    except Exception:
        return None


def generate_performance_report():
    """Generate a human-readable performance report for the baseline model."""
    metrics = calculate_model_metrics()
    if not metrics.get("artifact_path"):
        return """
     MODEL PERFORMANCE REPORT
    ==================================================

      No artifact-backed metrics were found.
    """

    report = f"""
     MODEL PERFORMANCE REPORT
    {'=' * 50}

     Model Information:
       Name: {metrics.get('model_name')}
       Type: {metrics.get('model_type')}
       Artifact: {metrics.get('artifact_path')}
       Features: {metrics.get('features_count')}

     Performance Metrics:
       R² Score: {metrics.get('r2_score', 0):.3f}
       Mean Absolute Error (MAE): {metrics.get('mean_absolute_error', 0):.1f} bikes
       Root Mean Squared Error (RMSE): {metrics.get('root_mean_squared_error', 0):.1f} bikes
       Mean Absolute Percentage Error: {metrics.get('mean_absolute_percentage_error', 'N/A')}

     Interpretation:
       {metrics.get('interpretation', {}).get('r2_interpretation', '')}
       {metrics.get('interpretation', {}).get('mae_interpretation', '')}
       {metrics.get('interpretation', {}).get('rmse_interpretation', '')}
    """

    top_features = metrics.get('top_features')
    if top_features:
        report += "\n     Top 5 Most Important Features:\n"
        for feat in top_features:
            report += f"      {feat['rank']}. {feat['feature']}: {feat['importance_percent']}\n"

    return report


def is_model_available():
    """Check if the baseline random forest model is available for predictions."""
    model_path = MODELS_DIR / 'random_forest_model.pkl'
    if not model_path.exists():
        return False
    try:
        joblib.load(model_path)
        return True
    except Exception:
        return False


if __name__ == '__main__':
    print("Testing model metrics module...")
    print(f"Model available: {is_model_available()}")
    print(generate_performance_report())
