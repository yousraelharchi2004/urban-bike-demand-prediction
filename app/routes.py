"""
Legacy compatibility shim.

The modular blueprints in ``app.blueprints`` are the active application surface.
This module intentionally stays small and only re-exports targeted helpers for
tests or external code that still imports ``app.routes``.
"""

from __future__ import annotations

from app.blueprints.analytics import date_range_analytics as analytics_date_range
from app.blueprints.export import (
    export_all_charts,
    export_chart,
    export_metrics_pdf,
    export_via_email,
    handle_export,
    send_analytics_email,
    test_email_config,
)
from app.blueprints.pages import (
    debug_advanced_feature_order,
    debug_improved_feature_order,
    debug_improved_model,
    debug_xgboost_feature_order,
    list_models,
    test_improved_model_prediction,
)
from app.blueprints.prediction import compare_predictions, predict_advanced, predict_xgboost_route
from app.services.input_normalization import get_required_fields_for_model, normalize_field_names_for_model

__all__ = [
    'analytics_date_range',
    'compare_predictions',
    'debug_advanced_feature_order',
    'debug_improved_feature_order',
    'debug_improved_model',
    'debug_xgboost_feature_order',
    'export_all_charts',
    'export_chart',
    'export_metrics_pdf',
    'export_via_email',
    'get_required_fields_for_model',
    'handle_export',
    'list_models',
    'normalize_field_names_for_model',
    'predict_advanced',
    'predict_xgboost_route',
    'send_analytics_email',
    'test_email_config',
    'test_improved_model_prediction',
]
