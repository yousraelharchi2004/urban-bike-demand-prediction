"""
Export Blueprint - Data export and email routes.
Routes: /export/*, /api/export/*
"""

from flask import Blueprint, request, jsonify, make_response
import json
import csv
import io
import os
from datetime import datetime

from app.utils import get_prediction_history

export = Blueprint('export', __name__)


@export.route('/export/predictions/csv', methods=['GET'])
def export_predictions_csv():
    """Export prediction history as CSV."""
    try:
        history = get_prediction_history(limit=None)

        output = io.StringIO()
        if history:
            fieldnames = list(history[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow({k: str(v) for k, v in row.items()})

        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=predictions_history.csv"
        response.headers["Content-Type"] = "text/csv"
        return response

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/export/predictions/json', methods=['GET'])
def export_predictions_json():
    """Export prediction history as JSON."""
    try:
        history = get_prediction_history(limit=None)

        response = make_response(json.dumps(history, indent=2, default=str))
        response.headers["Content-Disposition"] = "attachment; filename=predictions_history.json"
        response.headers["Content-Type"] = "application/json"
        return response

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@export.route('/export/model-report', methods=['GET'])
def export_model_report():
    """Export a model performance report as text."""
    try:
        from app.model_metrics import generate_performance_report
        report = generate_performance_report()

        response = make_response(report)
        response.headers["Content-Disposition"] = "attachment; filename=model_report.txt"
        response.headers["Content-Type"] = "text/plain"
        return response

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
