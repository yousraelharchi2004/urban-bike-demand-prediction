"""Prediction business logic shared by HTTP routes and compatibility layers."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.models.loader import get_state
from app.utils import (
    get_prediction_history,
    load_unified_pipeline,
    preprocess_input,
    preprocess_input_original,
    preprocess_unified,
    preprocess_with_advanced_features,
    save_prediction_to_history,
)


VALID_MODEL_TYPES = (
    'original',
    'improved',
    'advanced',
    'xgboost',
    'lightgbm',
    'stacking',
    'unified',
    'ensemble',
)
COMPARISON_MODEL_TYPES = (
    'original',
    'improved',
    'advanced',
    'xgboost',
    'lightgbm',
    'stacking',
    'unified',
)


def calculate_confidence(predicted_bikes: float, input_data: dict) -> float:
    """Calculate a simple confidence score (0-100) for a prediction."""
    confidence = 85.0

    if predicted_bikes > 3000 or predicted_bikes < 0:
        confidence -= 15
    elif predicted_bikes > 2000:
        confidence -= 5

    temp = float(input_data.get('Temperature(Â°C)', input_data.get('temperature', 20)))
    if temp < -10 or temp > 40:
        confidence -= 10
    elif temp < 0 or temp > 35:
        confidence -= 5

    rain = float(input_data.get('Rainfall(mm)', input_data.get('rainfall', 0)))
    if rain > 10:
        confidence -= 10

    return max(0, min(100, round(confidence, 1)))


def load_history(limit: int = 50) -> list[dict]:
    """Load prediction history from the canonical storage."""
    return get_prediction_history(limit=limit)


def _normalize_model_type(requested_model_type: str | None) -> str:
    model_type = (requested_model_type or 'original').lower()
    if model_type not in VALID_MODEL_TYPES:
        return 'original'
    return model_type


def _make_prediction(model: Any, features: Any, model_type: str) -> float:
    """Run inference for one model and clamp predictions to non-negative values."""
    if model_type == 'stacking' and hasattr(features, 'values'):
        features = features.values

    pred = model.predict(features)
    value = float(pred[0]) if isinstance(pred, np.ndarray) else float(pred)
    return max(0.0, value)


def _build_saved_prediction(input_data: dict, predicted: float, model_type: str, save_history: bool) -> dict | None:
    """Persist prediction history when requested."""
    if not save_history:
        return None

    entry = save_prediction_to_history(input_data, predicted, model_type)
    if entry:
        return entry

    return {
        'prediction_id': None,
        'timestamp': None,
        'confidence': calculate_confidence(predicted, input_data),
    }


def _unified_prediction_response(data: dict, save_history: bool) -> tuple[dict, int]:
    pipeline = load_unified_pipeline()
    if pipeline is None:
        return {"success": False, "error": "Unified pipeline not found"}, 404

    processed = preprocess_unified(data)
    predicted = max(0.0, float(pipeline['model'].predict(processed)[0]))
    saved_entry = _build_saved_prediction(data, predicted, 'unified', save_history)

    return {
        "success": True,
        "predicted_bikes": round(predicted),
        "predicted_bikes_raw": predicted,
        "model_type": "unified",
        "model_name": pipeline.get('model_name', 'stacking_ensemble'),
        "prediction_id": saved_entry.get('prediction_id') if saved_entry else None,
        "timestamp": saved_entry.get('timestamp') if saved_entry else None,
        "confidence": saved_entry.get('confidence', 0) if saved_entry else 0,
    }, 200


def _legacy_prediction_response(data: dict, model_type: str, state: dict, save_history: bool) -> tuple[dict, int]:
    model_key = f'{model_type}_model'
    model = state.get(model_key) or state['original_model']
    used_type = model_type if state.get(model_key) else 'original'

    try:
        if used_type == 'original':
            processed = preprocess_input_original(data)
        elif used_type in ('advanced', 'xgboost', 'lightgbm', 'stacking'):
            processed = preprocess_with_advanced_features(data, model=model)
        else:
            processed = preprocess_input(data, model_type=used_type, model=model)
    except Exception as exc:
        return {"success": False, "error": f"Preprocessing failed: {exc}"}, 400

    try:
        predicted = _make_prediction(model, processed, used_type)
    except Exception as exc:
        return {"success": False, "error": str(exc)}, 500

    saved_entry = _build_saved_prediction(data, predicted, used_type, save_history)
    return {
        "success": True,
        "predicted_bikes": round(predicted),
        "predicted_bikes_raw": predicted,
        "model_type": used_type,
        "prediction_id": saved_entry.get('prediction_id') if saved_entry else None,
        "timestamp": saved_entry.get('timestamp') if saved_entry else None,
        "confidence": saved_entry.get('confidence', 0) if saved_entry else 0,
    }, 200


def predict(data: dict | None, requested_model_type: str | None = None, save_history: bool = True) -> tuple[dict, int]:
    """Main prediction workflow independent of HTTP transport."""
    state = get_state()
    if not state['model_loaded'] or state['original_model'] is None:
        return {"success": False, "error": "Model not loaded"}, 503

    if not data:
        return {"success": False, "error": "No data provided"}, 400

    model_type = _normalize_model_type(requested_model_type)
    if model_type in ('unified', 'ensemble'):
        try:
            return _unified_prediction_response(data, save_history)
        except Exception as exc:
            return {"success": False, "error": str(exc)}, 500

    return _legacy_prediction_response(data, model_type, state, save_history)


def compare_predictions(data: dict | None) -> tuple[dict, int]:
    """Compare predictions across all supported model types."""
    if not data:
        return {"success": False, "error": "No data provided"}, 400

    results = {}
    compared = 0
    for model_type in COMPARISON_MODEL_TYPES:
        payload, status = predict(data, requested_model_type=model_type, save_history=False)
        if status == 200 and payload.get('success'):
            results[model_type] = {
                'prediction': payload.get('predicted_bikes_raw', payload.get('predicted_bikes', 0)),
                'rounded': payload.get('predicted_bikes', 0),
                'status': 'success',
            }
            compared += 1
        else:
            results[model_type] = {
                'status': 'error',
                'error': payload.get('error', f'Prediction failed with status {status}'),
            }

    return {
        "success": True,
        "results": results,
        "models_compared": compared,
    }, 200
