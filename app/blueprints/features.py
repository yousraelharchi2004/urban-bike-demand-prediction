"""
Features Blueprint - Feature engineering dashboard and API.
Routes: /feature-engineering, /api/features/*
"""

from flask import Blueprint, render_template, jsonify, request
import os
import json

from app.models.loader import get_state

features = Blueprint('features', __name__)


@features.route('/feature-engineering')
def feature_engineering():
    """Feature engineering dashboard page."""
    if os.path.exists('app/templates/feature_engineering.html'):
        return render_template('feature_engineering.html')
    return '<h1>Feature Engineering</h1><p>Dashboard coming soon.</p><a href="/">Home</a>'


@features.route('/api/features/advanced', methods=['GET'])
def get_advanced_features():
    """Return the list of features used by the advanced model."""
    state = get_state()
    feature_list = state.get('advanced_model_features', [])

    return jsonify({
        "success": True,
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
        temp = float(data.get('temperature', data.get('Temperature(°C)', 20)))
        humidity = float(data.get('humidity', data.get('Humidity(%)', 50)))

        calculated = {
            "hour_sin": round(float(np.sin(2 * np.pi * hour / 24)), 4),
            "hour_cos": round(float(np.cos(2 * np.pi * hour / 24)), 4),
            "hour_squared": hour ** 2,
            "temp_hour_interaction": temp * hour,
            "comfort_index": round(temp - humidity / 10, 2),
            "is_peak_hour": 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
        }

        return jsonify({"success": True, "calculated_features": calculated})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
