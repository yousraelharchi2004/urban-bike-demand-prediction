"""
Legacy compatibility routes.

This blueprint keeps only the legacy endpoints that do not yet have modular
replacements. Shared routes such as `/`, `/predict`, `/metrics`, and analytics
API endpoints are served by the modular blueprints and must not be duplicated
here.
"""

from flask import Blueprint

from app.blueprints.analytics import date_range_analytics as analytics_date_range_view
from app.blueprints.export import (
    export_all_charts as export_all_charts_view,
    export_chart as export_chart_view,
    export_metrics_pdf as export_metrics_pdf_view,
    export_via_email as export_via_email_view,
    handle_export as handle_export_view,
    send_analytics_email as send_analytics_email_view,
    test_email_config as test_email_config_view,
)
from app.blueprints.pages import (
    debug_advanced_feature_order as debug_advanced_feature_order_view,
    debug_improved_feature_order as debug_improved_feature_order_view,
    debug_improved_model as debug_improved_model_view,
    debug_xgboost_feature_order as debug_xgboost_feature_order_view,
    list_models as list_models_view,
    test_improved_model_prediction as test_improved_model_prediction_view,
)
from app.blueprints.prediction import (
    compare_predictions as compare_predictions_view,
    predict_advanced as predict_advanced_view,
    predict_xgboost_route as predict_xgboost_view,
)

legacy_compat = Blueprint("legacy_compat", __name__)


@legacy_compat.route("/debug/improved-feature-order", methods=["GET"])
def debug_improved_feature_order_route():
    return debug_improved_feature_order_view()


@legacy_compat.route("/debug/advanced-feature-order", methods=["GET"])
def debug_advanced_feature_order_route():
    return debug_advanced_feature_order_view()


@legacy_compat.route("/debug/xgboost-feature-order", methods=["GET"])
def debug_xgboost_feature_order_route():
    return debug_xgboost_feature_order_view()


@legacy_compat.route("/debug/improved-model", methods=["GET"])
def debug_improved_model_route():
    return debug_improved_model_view()


@legacy_compat.route("/test-improved-model", methods=["GET"])
def test_improved_model_prediction_route():
    return test_improved_model_prediction_view()


@legacy_compat.route("/predict/advanced", methods=["POST"])
def predict_advanced_route():
    return predict_advanced_view()


@legacy_compat.route("/predict/xgboost", methods=["POST"])
def predict_xgboost_route():
    return predict_xgboost_view()


@legacy_compat.route("/predict/compare", methods=["POST"])
def compare_predictions_route():
    return compare_predictions_view()


@legacy_compat.route("/api/models", methods=["GET"])
def list_models_route():
    return list_models_view()


@legacy_compat.route("/api/send-analytics-email", methods=["POST"])
def send_analytics_email_route():
    return send_analytics_email_view()


@legacy_compat.route("/api/export/chart/<chart_type>", methods=["GET"])
def export_chart_route(chart_type):
    return export_chart_view(chart_type)


@legacy_compat.route("/api/export/all-charts", methods=["GET"])
def export_all_charts_route():
    return export_all_charts_view()


@legacy_compat.route("/api/export/<export_type>", methods=["POST"])
def handle_export_route(export_type):
    return handle_export_view(export_type)


@legacy_compat.route("/export/metrics/pdf", methods=["GET"])
def export_metrics_pdf_route():
    return export_metrics_pdf_view()


@legacy_compat.route("/export/email", methods=["POST"])
def export_via_email_route():
    return export_via_email_view()


@legacy_compat.route("/export/email/test", methods=["GET"])
def test_email_config_route():
    return test_email_config_view()


@legacy_compat.route("/api/analytics/date-range", methods=["POST"])
def analytics_date_range_route():
    return analytics_date_range_view()
