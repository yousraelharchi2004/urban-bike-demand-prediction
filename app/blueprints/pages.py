"""
Pages Blueprint - Static page routes.
Routes: /, /test, /status, /metrics, /model-info
"""

from flask import Blueprint, render_template, jsonify, request
import os
import json
import pickle
from pathlib import Path

from app.models.loader import get_state
from app.model_metrics import calculate_model_metrics, generate_performance_report, get_model_catalog
from app.utils import get_prediction_history, preprocess_input
from app.artifact_contract import read_manifest, unified_manifest_path
from app.email_service import email_service

pages = Blueprint('pages', __name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json_if_exists(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _unified_available() -> bool:
    manifest = read_manifest(unified_manifest_path(_project_root()))
    return bool(manifest)


@pages.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')


@pages.route('/test')
def test():
    """Test page."""
    if os.path.exists('app/templates/test.html'):
        return render_template('test.html')
    return '<h1>Test Suite</h1><p>Test page coming soon.</p><a href="/">Home</a>'


@pages.route('/status')
def status():
    """Status dashboard page."""
    if os.path.exists('app/templates/status.html'):
        return render_template('status.html')
    return '<h1>System Status</h1><p>Status dashboard coming soon.</p><a href="/">Home</a>'


@pages.route('/api/status', methods=['GET'])
def api_status():
    """API status endpoint (JSON)."""
    state = get_state()
    try:
        # Use full history count (no default limit) for accurate status metrics.
        history = get_prediction_history(limit=None)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    models = {
        "original": {"loaded": state['original_model'] is not None},
        "improved": {"loaded": state['improved_model'] is not None},
        "advanced": {"loaded": state['advanced_model'] is not None},
        "xgboost": {"loaded": state['xgboost_model'] is not None},
        "lightgbm": {"loaded": state['lightgbm_model'] is not None},
        "stacking": {"loaded": state['stacking_model'] is not None},
        "unified": {"loaded": _unified_available()},
    }

    total_loaded = sum(1 for entry in models.values() if entry.get('loaded'))
    total_predictions = len(history)
    system_info = {
        "version": "2.1.4",
        "total_models_loaded": total_loaded,
        "models_loaded": total_loaded,
        "email_configured": email_service.configured,
    }

    return jsonify({
        "status": "success",
        "models": models,
        "total_predictions": total_predictions,
        "prediction_history": {
            "total_predictions": total_predictions
        },
        "system_info": system_info,
    })


@pages.route('/metrics', methods=['GET'])
def metrics():
    """Model performance metrics endpoint."""
    try:
        metrics_data = calculate_model_metrics()
        return jsonify({
            "success": True,
            "metrics": metrics_data,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@pages.route('/model-info', methods=['GET'])
def model_info():
    """Model information endpoint."""
    state = get_state()
    model_type = request.args.get('model_type', 'original')

    info = {"model_type": model_type, "available": False}

    model = state.get(f'{model_type}_model')
    if model is not None:
        info["available"] = True
        info["model_class"] = type(model).__name__
        if hasattr(model, 'n_features_in_'):
            info["features_count"] = int(model.n_features_in_)
        if hasattr(model, 'feature_names_in_'):
            info["feature_names"] = list(model.feature_names_in_)

    return jsonify(info)


@pages.route('/api/models', methods=['GET'])
def list_models():
    """List all supported models from the modular backend."""
    state = get_state()
    catalog = get_model_catalog()

    models = {
        'original': {
            'available': state['original_model'] is not None,
            **catalog['original'],
        },
        'improved': {
            'available': state['improved_model'] is not None,
            **catalog['improved'],
        },
        'advanced': {
            'available': state['advanced_model'] is not None,
            **catalog['advanced'],
            'auto_feature_calculation': True,
        },
        'xgboost': {
            'available': state['xgboost_model'] is not None,
            **catalog['xgboost'],
            'auto_feature_calculation': True,
        },
        'lightgbm': {
            'available': state['lightgbm_model'] is not None,
            **catalog['lightgbm'],
        },
        'stacking': {
            'available': state['stacking_model'] is not None,
            **catalog['stacking'],
        },
        'unified': {
            'available': _unified_available(),
            **catalog['unified'],
        },
    }

    # Override feature counts with loaded model metadata when available.
    # This avoids collapsing all models to unified pipeline feature count.
    state_model_keys = {
        'original': 'original_model',
        'improved': 'improved_model',
        'advanced': 'advanced_model',
        'xgboost': 'xgboost_model',
        'lightgbm': 'lightgbm_model',
        'stacking': 'stacking_model',
    }
    for model_key, state_key in state_model_keys.items():
        model_obj = state.get(state_key)
        if model_obj is None:
            continue
        feature_count = None
        if hasattr(model_obj, 'feature_names_in_'):
            try:
                feature_count = len(list(model_obj.feature_names_in_))
            except Exception:
                feature_count = None
        if feature_count is None and hasattr(model_obj, 'n_features_in_'):
            try:
                feature_count = int(model_obj.n_features_in_)
            except Exception:
                feature_count = None
        if feature_count is not None and model_key in models:
            models[model_key]['features'] = feature_count

    return jsonify({
        "success": True,
        "models": models,
        "default_model": "original",
        "advanced_features_available": state['advanced_model'] is not None,
        "xgboost_available": state['xgboost_model'] is not None,
        "feature_engineering_capabilities": {
            "time_features": "hour_sin, hour_cos, hour_squared",
            "interaction_features": "temp_hour_interaction, comfort_index",
            "indicator_features": "is_weekend, is_peak_hour, good_biking_conditions",
            "categorical_features": "temp_category, time_of_day with one-hot encoding",
        },
    })


def _feature_order_response(model, label):
    if model is None:
        return jsonify({"success": False, "error": f"{label} model not loaded"})

    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        booster = model.get_booster()
        features = booster.feature_names or []
    else:
        return jsonify({"success": False, "error": "Model does not expose feature names"})

    return jsonify({
        "success": True,
        "feature_count": len(features),
        "feature_order": features,
    })


@pages.route('/debug/improved-feature-order', methods=['GET'])
def debug_improved_feature_order():
    return _feature_order_response(get_state()['improved_model'], 'Improved')


@pages.route('/debug/advanced-feature-order', methods=['GET'])
def debug_advanced_feature_order():
    return _feature_order_response(get_state()['advanced_model'], 'Advanced')


@pages.route('/debug/xgboost-feature-order', methods=['GET'])
def debug_xgboost_feature_order():
    return _feature_order_response(get_state()['xgboost_model'], 'XGBoost')


@pages.route('/debug/improved-model', methods=['GET'])
def debug_improved_model():
    """Inspect the saved improved model artifact without legacy imports."""
    model_path = _project_root() / 'models' / 'improved_random_forest.pkl'
    if not model_path.exists():
        return jsonify({"success": False, "error": f"Improved model file not found at {model_path}"})

    try:
        with model_path.open('rb') as handle:
            model_data = pickle.load(handle)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    result = {
        "success": True,
        "model_path": str(model_path),
        "model_type": type(model_data).__name__,
        "is_dict": isinstance(model_data, dict),
    }

    if isinstance(model_data, dict):
        result["dict_keys"] = list(model_data.keys())
        for key, value in model_data.items():
            result[f"key_{key}_type"] = type(value).__name__
            result[f"key_{key}_has_predict"] = hasattr(value, 'predict')
            if hasattr(value, 'predict'):
                result["found_model_in_key"] = key
                result["model_found"] = True
                if hasattr(value, 'feature_names_in_'):
                    result["feature_names"] = list(value.feature_names_in_)[:10]
                break
    else:
        result["has_predict"] = hasattr(model_data, 'predict')
        result["model_found"] = hasattr(model_data, 'predict')

    return jsonify(result)


@pages.route('/test-improved-model', methods=['GET'])
def test_improved_model_prediction():
    """Smoke-test improved model preprocessing and prediction."""
    state = get_state()
    model = state['improved_model']
    if model is None:
        return jsonify({"success": False, "error": "Improved model not loaded"}), 503

    sample = {
        "Hour": 14,
        "Temperature(Â°C)": 22.5,
        "Humidity(%)": 60,
        "Wind speed (m/s)": 2.5,
        "Visibility (10m)": 2000,
        "Dew point temperature(Â°C)": 10,
        "Solar Radiation (MJ/m2)": 1.5,
        "Rainfall(mm)": 0,
        "Snowfall (cm)": 0,
        "day_of_week": 3,
    }

    try:
        processed = preprocess_input(sample, model_type='improved', model=model)
        prediction = float(model.predict(processed)[0])
        return jsonify({
            "success": True,
            "prediction": prediction,
            "rounded": round(prediction),
            "features_used": int(processed.shape[1]),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
