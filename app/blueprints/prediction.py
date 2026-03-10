"""
Prediction Blueprint - All prediction-related routes.
Routes: /predict, /predict/advanced, /predict/xgboost, /predict/compare, /prediction-history
"""

from flask import Blueprint, request, jsonify
import numpy as np
import json
import os

from app.models.loader import get_state
from app.utils import (
    preprocess_input,
    preprocess_input_original,
    preprocess_with_advanced_features,
    get_prediction_history,
    save_prediction_to_history,
    validate_input,
)

prediction = Blueprint('prediction', __name__)


def _make_prediction(model, features, model_type: str):
    """Common prediction logic."""
    pred = model.predict(features)
    if isinstance(pred, np.ndarray):
        value = float(pred[0])
    else:
        value = float(pred)
    return max(0, value)


@prediction.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint supporting multiple model types."""
    state = get_state()

    if not state['model_loaded'] or state['original_model'] is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        model_type = request.args.get('model_type', 'original').lower()
        if model_type not in ('original', 'improved', 'advanced', 'xgboost', 'lightgbm', 'stacking', 'unified', 'ensemble'):
            model_type = 'original'

        # Unified / ensemble path
        if model_type in ('unified', 'ensemble'):
            try:
                from app.utils import load_unified_pipeline, preprocess_unified
                pipeline = load_unified_pipeline()
                if pipeline is None:
                    return jsonify({"success": False, "error": "Unified pipeline not found"}), 404
                processed = preprocess_unified(data)
                pred = pipeline['model'].predict(processed)
                predicted = max(0, float(pred[0]))
                save_prediction_to_history(data, predicted, 'unified')
                return jsonify({
                    "success": True,
                    "predicted_bikes": round(predicted),
                    "predicted_bikes_raw": predicted,
                    "model_type": "unified_ensemble",
                    "model_name": pipeline.get('model_name', 'stacking_ensemble'),
                })
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        # Legacy model path
        model_key = f'{model_type}_model'
        model = state.get(model_key) or state['original_model']
        used_type = model_type if state.get(model_key) else 'original'

        try:
            if used_type == 'original':
                processed = preprocess_input_original(data)
            elif used_type in ('advanced', 'xgboost', 'lightgbm', 'stacking'):
                processed = preprocess_with_advanced_features(data, model_type=used_type)
            else:
                processed = preprocess_input(data, model_type=used_type)
        except Exception as e:
            return jsonify({"success": False, "error": f"Preprocessing failed: {e}"}), 400

        predicted = _make_prediction(model, processed, used_type)
        save_prediction_to_history(data, predicted, used_type)

        return jsonify({
            "success": True,
            "predicted_bikes": round(predicted),
            "predicted_bikes_raw": predicted,
            "model_type": used_type,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@prediction.route('/prediction-history', methods=['GET'])
def prediction_history():
    """Return prediction history."""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = get_prediction_history(limit=limit)
        return jsonify({
            "success": True,
            "predictions": history,
            "count": len(history),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
