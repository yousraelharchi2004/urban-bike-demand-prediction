"""
Pages Blueprint - Static page routes.
Routes: /, /test, /status, /metrics, /model-info
"""

from flask import Blueprint, render_template, jsonify, request
import os
import json

from app.models.loader import get_state
from app.model_metrics import calculate_model_metrics, generate_performance_report
from app.utils import get_prediction_history

pages = Blueprint('pages', __name__)


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
        history = get_prediction_history()
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    return jsonify({
        "status": "success",
        "models": {
            "original": {"loaded": state['original_model'] is not None},
            "improved": {"loaded": state['improved_model'] is not None},
            "advanced": {"loaded": state['advanced_model'] is not None},
            "xgboost": {"loaded": state['xgboost_model'] is not None},
        },
        "total_predictions": len(history),
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
