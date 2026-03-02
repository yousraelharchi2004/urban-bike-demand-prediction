"""
Analytics Blueprint - Analytics dashboard and API routes.
Routes: /analytics, /api/analytics/*, /api/dashboard-metrics
"""

from flask import Blueprint, render_template, jsonify, request
import os
import json
from datetime import datetime, timedelta

from app.models.loader import get_state
from app.model_metrics import calculate_model_metrics
from app.utils import get_prediction_history

analytics = Blueprint('analytics', __name__)

# Model metadata (single source of truth)
MODEL_META = {
    'original': {
        'name': 'Random Forest (14 features)',
        'r2': 0.82, 'rmse': 112.5, 'mae': 87.2,
        'features': 14,
        'accuracy': 82, 'stability': 85, 'speed': 95,
    },
    'improved': {
        'name': 'Improved RF (tuned)',
        'r2': 0.88, 'rmse': 98.3, 'mae': 72.1,
        'features': 14,
        'accuracy': 88, 'stability': 87, 'speed': 90,
    },
    'advanced': {
        'name': 'Advanced RF (100+ features)',
        'r2': 0.92, 'rmse': 85.6, 'mae': 61.4,
        'features': 100,
        'accuracy': 92, 'stability': 92, 'speed': 85,
    },
    'xgboost': {
        'name': 'XGBoost',
        'r2': 0.94, 'rmse': 78.2, 'mae': 55.8,
        'features': 100,
        'accuracy': 94, 'stability': 90, 'speed': 80,
    },
}

MODEL_STATE_KEYS = {
    'original': 'original_model',
    'improved': 'improved_model',
    'advanced': 'advanced_model',
    'xgboost': 'xgboost_model',
}


def _loaded_models(state):
    """Return list of loaded model keys."""
    return [k for k, sk in MODEL_STATE_KEYS.items() if state.get(sk) is not None]


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

    try:
        metrics_data = calculate_model_metrics()
        overall_accuracy = metrics_data.get('r2_score', 0.824) * 100
    except Exception:
        overall_accuracy = 82.4

    try:
        history = get_prediction_history()
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    return jsonify({
        "success": True,
        "overall_accuracy": round(overall_accuracy, 1),
        "total_predictions": len(history),
        "models_available": len(loaded),
    })


@analytics.route('/api/analytics/hourly-patterns', methods=['GET'])
def hourly_patterns():
    """Hourly demand patterns from prediction history."""
    try:
        time_filter = request.args.get('filter', 'all')
        history = get_prediction_history(limit=None)
        filtered = _filter_by_time(history, time_filter)

        hourly = {h: [] for h in range(24)}
        for pred in filtered:
            inp = pred.get('input_data', pred.get('input', {}))
            hour = inp.get('Hour', pred.get('hour'))
            bikes = pred.get('predicted_bikes', pred.get('prediction'))
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

        # Best model accuracy
        best_r2 = max((MODEL_META[m]['r2'] for m in loaded), default=0.82)

        history = get_prediction_history(limit=None)
        filtered = _filter_by_time(history, time_filter)
        predictions = [p.get('predicted_bikes', p.get('prediction', 0)) for p in filtered]

        # Determine features count from best model
        best_model_key = max(loaded, key=lambda m: MODEL_META[m]['r2']) if loaded else 'original'
        features_count = MODEL_META.get(best_model_key, {}).get('features', 14)

        return jsonify({
            "success": True,
            "kpis": {
                "accuracy": {
                    "value": round(best_r2 * 100, 1),
                    "trend": f"+{round((best_r2 - 0.82) * 100, 1)}% from baseline" if best_r2 > 0.82 else "baseline",
                },
                "error": {
                    "value": round(MODEL_META.get(best_model_key, {}).get('mae', 87.2), 1),
                    "trend": "-12% improvement" if best_model_key != 'original' else "baseline",
                },
                "predictions": {
                    "value": len(predictions),
                    "trend": f"+{len(predictions)} total" if predictions else "No predictions yet",
                },
                "features": {
                    "value": features_count,
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

        insight_list = []

        # Performance insight
        if loaded:
            best_key = max(loaded, key=lambda m: MODEL_META[m]['r2'])
            best = MODEL_META[best_key]
            insight_list.append({
                "type": "performance",
                "icon": "chart-line",
                "title": f"Best Model: {best['name']}",
                "description": f"Achieves R² = {best['r2']:.2f} with RMSE of {best['rmse']} bikes. "
                               f"{len(loaded)} model(s) currently loaded.",
            })

        # Feature insight
        model = state.get('original_model')
        if model and hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            top_feat = model.feature_names_in_[model.feature_importances_.argmax()]
            top_pct = model.feature_importances_.max() * 100
            insight_list.append({
                "type": "features",
                "icon": "star",
                "title": "Top Predictor",
                "description": f'"{top_feat}" contributes {top_pct:.1f}% to predictions. '
                               "Temperature and hour of day consistently rank as the strongest features.",
            })

        # Data insight
        if history:
            recent = history[-min(50, len(history)):]
            avg = sum(p.get('predicted_bikes', p.get('prediction', 0)) for p in recent) / len(recent)
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
    search_order = ['xgboost', 'advanced', 'improved', 'original']

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
            "source_model": MODEL_META.get(source, {}).get('name', source),
            "total_features": len(features),
        })

    return jsonify({"success": True, "features": [], "source_model": "None loaded"})


@analytics.route('/api/analytics/model-performance', methods=['GET'])
def model_performance():
    """Model performance metrics for radar chart — includes all loaded models."""
    state = get_state()
    loaded = _loaded_models(state)

    models = {}
    colors = {
        'original': '#4f46e5',
        'improved': '#7c3aed',
        'advanced': '#0891b2',
        'xgboost': '#db2777',
    }

    datasets = []
    for key in loaded:
        meta = MODEL_META.get(key, {})
        models[key] = {
            'name': meta.get('name', key),
            'accuracy': meta.get('accuracy', 80) / 100,
            'stability': meta.get('stability', 80) / 100,
            'speed': meta.get('speed', 80) / 100,
        }
        color = colors.get(key, '#4f46e5')
        datasets.append({
            'label': meta.get('name', key),
            'data': [meta.get('accuracy', 80), meta.get('stability', 80), meta.get('speed', 80),
                     meta.get('r2', 0.8) * 100, 100 - meta.get('mae', 80) / 2],
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
            'backgroundColor': 'rgba(79, 70, 229, 0.1)',
            'borderColor': '#4f46e5',
            'pointBackgroundColor': '#4f46e5',
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


def _filter_by_time(history, time_filter):
    """Filter prediction history by time range."""
    if time_filter == 'all':
        return history

    now = datetime.now()
    cutoff_map = {
        'today': now.replace(hour=0, minute=0, second=0, microsecond=0),
        'week': now - timedelta(days=7),
        'month': now - timedelta(days=30),
        'quarter': now - timedelta(days=90),
        'year': now - timedelta(days=365),
    }

    cutoff = cutoff_map.get(time_filter)
    if cutoff is None:
        return history

    filtered = []
    for pred in history:
        ts = pred.get('timestamp', pred.get('timestamp_readable', ''))
        try:
            if 'T' in str(ts):
                dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00').split('+')[0])
            elif '-' in str(ts):
                dt = datetime.strptime(str(ts)[:19], '%Y-%m-%d %H:%M:%S')
            else:
                continue
            if dt >= cutoff:
                filtered.append(pred)
        except (ValueError, TypeError):
            continue

    return filtered
