"""
Features Blueprint - Feature engineering dashboard and API.
Routes: /feature-engineering, /api/features/*
"""

from flask import Blueprint, render_template, jsonify, request
import os
import json

from app.models.loader import get_state
from app.utils import load_unified_feature_names

features = Blueprint('features', __name__)


@features.route('/feature-engineering')
def feature_engineering():
    """Feature engineering dashboard page."""
    if os.path.exists('app/templates/feature_engineering.html'):
        return render_template('feature_engineering.html')
    return '<h1>Feature Engineering</h1><p>Dashboard coming soon.</p><a href="/">Home</a>'


@features.route('/api/features/advanced', methods=['GET'])
def get_advanced_features():
    """Return feature names for the requested model (default: advanced)."""
    state = get_state()
    model_type = (request.args.get('model_type') or 'advanced').strip().lower()

    key_map = {
        'original': 'original_model',
        'improved': 'improved_model',
        'advanced': 'advanced_model',
        'xgboost': 'xgboost_model',
        'lightgbm': 'lightgbm_model',
        'stacking': 'stacking_model',
        'unified': None,
    }
    model_key = key_map.get(model_type, 'advanced_model')
    model = state.get(model_key) if model_key else None

    feature_list = []
    if model_type == 'unified':
        feature_list = load_unified_feature_names() or []
    if model is not None and hasattr(model, 'feature_names_in_'):
        try:
            feature_list = list(model.feature_names_in_)
        except Exception:
            feature_list = []

    if not feature_list and model_type == 'advanced':
        feature_list = state.get('advanced_model_features', [])
    if not feature_list and model_type == 'xgboost':
        feature_list = state.get('xgboost_model_features', [])

    return jsonify({
        "success": True,
        "model_type": model_type,
        "features": feature_list,
        "count": len(feature_list),
    })


@features.route('/api/features/importance', methods=['GET'])
def feature_importance():
    """Feature importance for the requested model."""
    state = get_state()
    model_type = request.args.get('model_type', 'original')
    model = state.get(f'{model_type}_model') or state.get('original_model')

    if model and hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
        names = list(model.feature_names_in_)
        importances = model.feature_importances_.tolist()
        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return jsonify({
            "success": True,
            "model_type": model_type,
            "features": [{"name": n, "importance": round(v, 4)} for n, v in pairs],
        })

    return jsonify({"success": True, "features": [], "message": "No feature importance available"})


@features.route('/api/features/calculate', methods=['POST'])
def calculate_features():
    """Calculate engineered features from basic inputs (for UI preview)."""
    import numpy as np

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    try:
        hour = float(data.get('Hour', data.get('hour', 12)))
        temp = float(data.get('temperature', data.get('Temperature(°C)', data.get('Temperature(Â°C)', 20))))
        humidity = float(data.get('humidity', data.get('Humidity(%)', 50)))
        season_raw = data.get('Seasons', data.get('season', 'Summer'))
        season = str(season_raw).strip().lower()

        season_spring = 1 if season == 'spring' else 0
        season_summer = 1 if season == 'summer' else 0
        season_autumn = 1 if season in ('autumn', 'fall') else 0
        season_winter = 1 if season == 'winter' else 0

        calculated = {
            "hour_sin": round(float(np.sin(2 * np.pi * hour / 24)), 4),
            "hour_cos": round(float(np.cos(2 * np.pi * hour / 24)), 4),
            "hour_squared": round(hour ** 2, 4),
            "temp_hour_interaction": round(temp * hour, 4),
            "temp_humidity_interaction": round(temp * humidity, 4),
            "humidity_squared": round(humidity ** 2, 4),
            "comfort_index": round(temp - humidity / 10, 4),
            "is_peak_hour": 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
            "season_Spring": season_spring,
            "season_Summer": season_summer,
            "season_Autumn": season_autumn,
            "season_Winter": season_winter,
        }

        return jsonify({"success": True, "calculated_features": calculated})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
