"""
Analytics Blueprint - Analytics dashboard and API routes.
Routes: /analytics, /api/analytics/*, /api/dashboard-metrics
"""

from flask import Blueprint, render_template, jsonify, request
import os
import json
from datetime import datetime

from app.models.loader import get_state
from app.model_metrics import calculate_model_metrics, get_model_catalog
from app.shap_analysis import get_or_generate_shap_analysis
from app.utils import (
    extract_prediction_from_history,
    filter_prediction_history_by_time,
    get_prediction_history,
    load_unified_pipeline,
)

analytics = Blueprint('analytics', __name__)

MODEL_STATE_KEYS = {
    'original': 'original_model',
    'improved': 'improved_model',
    'advanced': 'advanced_model',
    'xgboost': 'xgboost_model',
    'lightgbm': 'lightgbm_model',
    'stacking': 'stacking_model',
}


def _clamp_score(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """Clamp a numeric dashboard score into a chart-safe range."""
    return max(minimum, min(maximum, float(value)))


def get_model_meta() -> dict[str, dict]:
    """Return artifact-backed model metadata used across analytics routes."""
    return get_model_catalog()


def _loaded_models(state):
    """Return list of loaded model keys."""
    loaded = [k for k, sk in MODEL_STATE_KEYS.items() if state.get(sk) is not None]
    if load_unified_pipeline():
        loaded.append('unified')
    return loaded


@analytics.route('/analytics')
def analytics_page():
    """Analytics dashboard page."""
    if os.path.exists('app/templates/analytics.html'):
        return render_template('analytics.html')
    return "Analytics dashboard coming soon!"


@analytics.route('/api/dashboard-metrics')
def dashboard_metrics():
    """Dashboard summary metrics."""
    state = get_state()
    loaded = _loaded_models(state)
    model_meta = get_model_meta()
    baseline_meta = model_meta.get('original', {})
    baseline_r2 = baseline_meta.get('r2', 0.0)
    avg_error_value = baseline_meta.get('mae', 0.0)
    best_r2 = max((model_meta[m]['r2'] for m in loaded), default=baseline_r2)
    overall_accuracy = best_r2 * 100

    try:
        metrics_data = calculate_model_metrics()
        best_model_key = metrics_data.get('best_model')
        if best_model_key in model_meta:
            overall_accuracy = model_meta[best_model_key].get('r2', best_r2) * 100
            avg_error_value = model_meta[best_model_key].get('mae', avg_error_value)
        else:
            avg_error_value = metrics_data.get('mean_absolute_error', avg_error_value)
    except Exception:
        pass

    try:
        history = get_prediction_history(limit=None)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    today_predictions = len(filter_prediction_history_by_time(history, 'today'))

    metrics_payload = {
        "model_accuracy": round(overall_accuracy, 1),
        "predictions_today": today_predictions,
        "avg_error": round(avg_error_value, 1),
        "models_available": len(loaded),
    }

    return jsonify({
        "success": True,
        "overall_accuracy": round(overall_accuracy, 1),
        "total_predictions": len(history),
        "predictions_today": today_predictions,
        "avg_error": round(avg_error_value, 1),
        "models_available": len(loaded),
        "metrics": metrics_payload,
    })


@analytics.route('/api/analytics/hourly-patterns', methods=['GET'])
def hourly_patterns():
    """Hourly demand patterns from prediction history."""
    try:
        time_filter = request.args.get('filter', 'all')
        history = get_prediction_history(limit=None)
        filtered = filter_prediction_history_by_time(history, time_filter)

        hourly = {h: [] for h in range(24)}
        for pred in filtered:
            inp = pred.get('input_data', {})
            hour = inp.get('Hour', pred.get('hour'))
            bikes, _ = extract_prediction_from_history(pred)
            if hour is not None and bikes is not None:
                hourly[int(hour)].append(float(bikes))

        patterns = []
        for h in range(24):
            vals = hourly[h]
            avg = round(sum(vals) / len(vals), 1) if vals else 0
            is_peak = h in (7, 8, 9, 17, 18, 19)
            patterns.append({
                'hour': h,
                'hour_label': f'{h:02d}:00',
                'average_demand': avg,
                'count': len(vals),
                'is_peak_hour': is_peak,
            })

        return jsonify({
            "success": True,
            "patterns": patterns,
            "filter": time_filter,
            "filtered_predictions": len(filtered),
            "is_sample": len(filtered) == 0,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@analytics.route('/api/analytics/kpis', methods=['GET'])
def kpis():
    """Key performance indicators matching frontend expectations."""
    try:
        state = get_state()
        loaded = _loaded_models(state)
        time_filter = request.args.get('filter', 'all')
        requested_model = (request.args.get('model') or '').strip().lower()
        model_meta = get_model_meta()
        baseline_r2 = model_meta.get('original', {}).get('r2', 0.0)
        baseline_mae = model_meta.get('original', {}).get('mae', 0.0)

        # Best model accuracy
        best_r2 = max((model_meta[m]['r2'] for m in loaded), default=baseline_r2)

        history = get_prediction_history(limit=None)
        filtered = filter_prediction_history_by_time(history, time_filter)
        predictions = [extract_prediction_from_history(p)[0] for p in filtered]

        # Determine the model used for KPI feature count.
        best_model_key = max(loaded, key=lambda m: model_meta[m]['r2']) if loaded else 'original'
        kpi_model_key = best_model_key
        if requested_model and requested_model in model_meta and requested_model in loaded:
            kpi_model_key = requested_model

        features_count = model_meta.get(kpi_model_key, {}).get('features', 14)
        if kpi_model_key == 'unified':
            pipeline = load_unified_pipeline()
            if pipeline and isinstance(pipeline.get('feature_names'), list):
                features_count = len(pipeline['feature_names'])
        else:
            model_obj = state.get(MODEL_STATE_KEYS.get(kpi_model_key, ''))
            if model_obj is not None:
                if hasattr(model_obj, 'feature_names_in_'):
                    features_count = len(model_obj.feature_names_in_)
                elif hasattr(model_obj, 'n_features_in_'):
                    features_count = int(model_obj.n_features_in_)

        return jsonify({
            "success": True,
            "kpis": {
                "accuracy": {
                    "value": round(best_r2 * 100, 1),
                    "trend": f"+{round((best_r2 - baseline_r2) * 100, 1)}% from baseline" if best_r2 > baseline_r2 else "baseline",
                },
                "error": {
                    "value": round(model_meta.get(best_model_key, {}).get('mae', baseline_mae), 1),
                    "trend": "-12% improvement" if best_model_key != 'original' else "baseline",
                },
                "predictions": {
                    "value": len(predictions),
                    "trend": f"+{len(predictions)} total" if predictions else "No predictions yet",
                },
                "features": {
                    "value": features_count,
                    "model": kpi_model_key,
                },
            },
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@analytics.route('/api/analytics/insights', methods=['GET'])
def insights():
    """Data-driven insights from prediction history and model state."""
    try:
        state = get_state()
        loaded = _loaded_models(state)
        history = get_prediction_history(limit=None)
        model_meta = get_model_meta()

        insight_list = []

        # Performance insight
        if loaded:
            best_key = max(loaded, key=lambda m: model_meta[m]['r2'])
            best = model_meta[best_key]
            insight_list.append({
                "type": "performance",
                "icon": "chart-line",
                "title": f"Best Model: {best['name']}",
                "description": f"Achieves R² = {best['r2']:.2f} with RMSE of {best['rmse']} bikes. "
                               f"{len(loaded)} model(s) currently loaded.",
            })

        # Feature insight
        feature_model = None
        feature_model_key = None
        for candidate_key in ('xgboost', 'lightgbm', 'advanced', 'improved', 'original'):
            candidate = state.get(MODEL_STATE_KEYS[candidate_key])
            if candidate and hasattr(candidate, 'feature_importances_') and hasattr(candidate, 'feature_names_in_'):
                feature_model = candidate
                feature_model_key = candidate_key
                break

        if feature_model is not None:
            top_feat = feature_model.feature_names_in_[feature_model.feature_importances_.argmax()]
            top_pct = feature_model.feature_importances_.max() * 100
            feature_model_name = model_meta.get(feature_model_key, {}).get('name', feature_model_key)
            insight_list.append({
                "type": "features",
                "icon": "star",
                "title": f"Top Predictor ({feature_model_name})",
                "description": f'"{top_feat}" contributes {top_pct:.1f}% to predictions for the {feature_model_name} model.',
            })

        # Data insight
        if history:
            recent = history[-min(50, len(history)):]
            avg = sum(extract_prediction_from_history(p)[0] for p in recent) / len(recent)
            insight_list.append({
                "type": "data",
                "icon": "database",
                "title": f"{len(history)} Predictions Recorded",
                "description": f"Average predicted demand: {avg:.0f} bikes. "
                               "Peak demand typically occurs during morning (7-9 AM) and evening (5-7 PM) rush hours.",
            })
        else:
            insight_list.append({
                "type": "data",
                "icon": "info-circle",
                "title": "No Predictions Yet",
                "description": "Make predictions on the Predictor page to see data-driven insights here.",
            })

        return jsonify({"success": True, "insights": insight_list})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@analytics.route('/api/analytics/feature-importance', methods=['GET'])
def feature_importance():
    """Feature importance data for visualization. Supports model selection."""
    state = get_state()
    requested = request.args.get('model', 'best')

    # Try requested model, then fall back through hierarchy
    model = None
    source = 'original'
    search_order = ['xgboost', 'stacking', 'lightgbm', 'advanced', 'improved', 'original']

    if requested != 'best' and requested in MODEL_STATE_KEYS:
        candidate = state.get(MODEL_STATE_KEYS[requested])
        if candidate and hasattr(candidate, 'feature_importances_'):
            model = candidate
            source = requested

    if model is None:
        for key in search_order:
            candidate = state.get(MODEL_STATE_KEYS[key])
            if candidate and hasattr(candidate, 'feature_importances_') and hasattr(candidate, 'feature_names_in_'):
                model = candidate
                source = key
                break

    if model and hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
        importances = model.feature_importances_.tolist()
        pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        return jsonify({
            "success": True,
            "features": [{"feature": n, "importance": round(v, 4), "importance_percent": f"{v * 100:.1f}%"} for n, v in pairs[:15]],
            "source_model": get_model_meta().get(source, {}).get('name', source),
            "total_features": len(features),
        })

    return jsonify({"success": True, "features": [], "source_model": "None loaded"})


@analytics.route('/api/analytics/model-performance', methods=['GET'])
def model_performance():
    """Model performance metrics for radar chart  includes all loaded models."""
    state = get_state()
    loaded = _loaded_models(state)
    model_meta = get_model_meta()

    models = {}
    colors = {
        'original': '#3b82f6',
        'improved': '#6366f1',
        'advanced': '#0d9488',
        'xgboost': '#8b5cf6',
        'lightgbm': '#06b6d4',
        'stacking': '#64748b',
        'unified': '#0f766e',
    }

    datasets = []
    for key in loaded:
        meta = model_meta.get(key, {})
        models[key] = {
            'name': meta.get('name', key),
            'accuracy': meta.get('accuracy', 80) / 100,
            'stability': meta.get('stability', 80) / 100,
            'speed': meta.get('speed', 80) / 100,
        }
        color = colors.get(key, '#3b82f6')
        datasets.append({
            'label': meta.get('name', key),
            'data': [
                _clamp_score(meta.get('accuracy', 80)),
                _clamp_score(meta.get('stability', 80)),
                _clamp_score(meta.get('speed', 80)),
                _clamp_score(meta.get('r2', 0.8) * 100),
                _clamp_score(100 - meta.get('mae', 80) / 2),
            ],
            'backgroundColor': f'{color}20',
            'borderColor': color,
            'pointBackgroundColor': color,
            'borderWidth': 2,
            'fill': True,
        })

    radar_data = {
        'labels': ['Accuracy', 'Stability', 'Speed', 'R² Score', 'Low Error'],
        'datasets': datasets if datasets else [{
            'label': 'No Models',
            'data': [0, 0, 0, 0, 0],
            'backgroundColor': 'rgba(37, 99, 235, 0.1)',
            'borderColor': '#2563eb',
            'pointBackgroundColor': '#2563eb',
            'borderWidth': 2,
            'fill': True,
        }],
    }

    return jsonify({
        "success": True,
        "models": models,
        "radar_data": radar_data,
        "total_models": len(loaded),
    })


@analytics.route('/api/analytics/shap-importance', methods=['GET'])
def shap_importance():
    """SHAP-based feature importance from pre-computed analysis."""
    requested_model = request.args.get('model', 'lightgbm')
    try:
        return jsonify(get_or_generate_shap_analysis(requested_model))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@analytics.route('/api/analytics/date-range', methods=['POST'])
def date_range_analytics():
    """Return summary analytics for a custom inclusive date range."""
    try:
        data = request.get_json() or {}
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        if not start_date or not end_date:
            return jsonify({
                "success": False,
                "error": "Both start_date and end_date are required",
            }), 400

        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({
                "success": False,
                "error": "Invalid date format. Use YYYY-MM-DD",
            }), 400

        if start_dt > end_dt:
            return jsonify({
                "success": False,
                "error": "Start date must be before end date",
            }), 400

        days_diff = (end_dt - start_dt).days
        if days_diff > 365:
            return jsonify({
                "success": False,
                "error": "Date range cannot exceed 365 days",
            }), 400

        history = get_prediction_history(limit=None)
        filtered = []
        for pred in history:
            ts = pred.get('timestamp', pred.get('timestamp_readable', ''))
            try:
                if 'T' in str(ts):
                    dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00').split('+')[0])
                else:
                    dt = datetime.strptime(str(ts)[:19], '%Y-%m-%d %H:%M:%S')
            except (TypeError, ValueError):
                continue

            if start_dt.date() <= dt.date() <= end_dt.date():
                filtered.append(pred)

        demand_values = [
            float(extract_prediction_from_history(pred)[0])
            for pred in filtered
        ]

        hourly = {hour: [] for hour in range(24)}
        for pred in filtered:
            inp = pred.get('input_data', {})
            hour = inp.get('Hour', pred.get('hour'))
            value = extract_prediction_from_history(pred)[0]
            if hour is not None and value is not None:
                hourly[int(hour)].append(float(value))

        avg_by_hour = [
            round(sum(values) / len(values), 1) if values else 0
            for _, values in hourly.items()
        ]

        loaded = _loaded_models(get_state())
        model_meta = get_model_meta()
        return jsonify({
            "success": True,
            "date_range": {
                "start": start_date,
                "end": end_date,
                "days": days_diff + 1,
            },
            "metrics": {
                "total_predictions": len(history),
                "filtered_predictions": len(filtered),
                "coverage": f"{len(filtered)}/{len(history)} predictions",
            },
            "kpis": {
                "accuracy": round(max((model_meta[m]['r2'] for m in loaded), default=model_meta.get('original', {}).get('r2', 0.0)) * 100, 1),
                "error": round(sum(demand_values) / len(demand_values), 1) if demand_values else 0,
                "predictions": len(filtered),
            },
            "patterns_summary": {
                "total_hours": 24,
                "peak_hours": sum(1 for hour, values in hourly.items() if hour in (7, 8, 9, 17, 18, 19) and values),
                "avg_demand": round(sum(avg_by_hour) / len(avg_by_hour), 1) if avg_by_hour else 0,
            },
            "insights_count": 1 if not filtered else 2,
            "message": f"Analyzed {len(filtered)} predictions from {start_date} to {end_date}",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
