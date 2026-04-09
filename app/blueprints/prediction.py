"""
Prediction Blueprint - All prediction-related routes.
Routes: /predict, /predict/advanced, /predict/xgboost, /predict/compare, /prediction-history
"""

from flask import Blueprint, request, jsonify

from app.services import prediction_service
from app.utils import filter_prediction_history_by_time

prediction = Blueprint('prediction', __name__)


def _get_json_payload():
    """Validate request content type and return parsed JSON payload."""
    if not request.is_json:
        return None, ({"success": False, "error": "Content-Type must be application/json"}, 415)
    return request.get_json(silent=True), None


@prediction.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint supporting multiple model types."""
    try:
        data, error = _get_json_payload()
        if error:
            payload, status = error
            return jsonify(payload), status
        payload, status = prediction_service.predict(
            data,
            requested_model_type=request.args.get('model_type'),
        )
        return jsonify(payload), status
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@prediction.route('/predict/advanced', methods=['POST'])
def predict_advanced():
    """Compatibility alias for advanced model predictions."""
    data, error = _get_json_payload()
    if error:
        payload, status = error
        return jsonify(payload), status
    payload, status = prediction_service.predict(data, requested_model_type='advanced')
    return jsonify(payload), status


@prediction.route('/predict/xgboost', methods=['POST'])
def predict_xgboost_route():
    """Compatibility alias for XGBoost predictions."""
    data, error = _get_json_payload()
    if error:
        payload, status = error
        return jsonify(payload), status
    payload, status = prediction_service.predict(data, requested_model_type='xgboost')
    return jsonify(payload), status


@prediction.route('/predict/compare', methods=['POST'])
def compare_predictions():
    """Compare predictions across all supported model types."""
    data, error = _get_json_payload()
    if error:
        payload, status = error
        return jsonify(payload), status
    payload, status = prediction_service.compare_predictions(data)
    return jsonify(payload), status


@prediction.route('/prediction-history', methods=['GET'])
def prediction_history():
    """Return prediction history."""
    try:
        limit = request.args.get('limit', 50, type=int)
        if limit is not None and limit <= 0:
            limit = None
        time_filter = request.args.get('filter', 'all')

        full_history = prediction_service.load_history(limit=None)
        filtered_history = filter_prediction_history_by_time(full_history, time_filter)
        displayed_history = filtered_history if limit is None else filtered_history[-limit:]

        return jsonify({
            "success": True,
            "history": displayed_history,
            "predictions": displayed_history,
            "filtered_count": len(filtered_history),
            "total_predictions": len(full_history),
            "filter": time_filter,
            "count": len(displayed_history),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
