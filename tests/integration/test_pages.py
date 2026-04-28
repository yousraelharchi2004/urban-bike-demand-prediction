"""
Integration tests for page and status endpoints.
"""

import pytest
import os
import sys
import json
from pathlib import Path
from app.model_metrics import get_model_catalog

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestPageRoutes:
    """Test that page routes return 200."""

    def test_home_page(self, client):
        """GET / should return 200."""
        response = client.get('/')
        assert response.status_code == 200

    def test_test_page(self, client):
        """GET /test should return 200."""
        response = client.get('/test')
        assert response.status_code == 200

    def test_status_page(self, client):
        """GET /status should return 200."""
        response = client.get('/status')
        assert response.status_code == 200

    def test_analytics_page(self, client):
        """GET /analytics should return 200."""
        response = client.get('/analytics')
        assert response.status_code == 200

    def test_feature_engineering_page(self, client):
        """GET /feature-engineering should return 200."""
        response = client.get('/feature-engineering')
        assert response.status_code == 200


class TestAPIStatusEndpoint:
    """Test the API status JSON endpoint."""

    def test_api_status_returns_json(self, client):
        """GET /api/status should return valid JSON."""
        response = client.get('/api/status')
        assert response.status_code == 200
        data = response.get_json()
        assert data is not None

    def test_api_status_has_models_info(self, client):
        """API status should include model information."""
        response = client.get('/api/status')
        data = response.get_json()
        if data:
            # Should have some model info
            assert 'models' in data or 'status' in data


class TestMetricsEndpoint:
    """Test the metrics endpoint."""

    def test_metrics_returns_200(self, client):
        """GET /metrics should return 200."""
        response = client.get('/metrics')
        assert response.status_code == 200

    def test_metrics_returns_json(self, client):
        """GET /metrics should return valid JSON."""
        response = client.get('/metrics')
        data = response.get_json()
        assert data is not None

    def test_metrics_uses_artifact_values(self, client):
        """GET /metrics should reflect artifact-backed baseline values."""
        response = client.get('/metrics')
        data = response.get_json()
        assert response.status_code == 200
        assert data['success'] is True

        artifact = get_model_catalog()['original']
        metrics = data['metrics']
        assert metrics['r2_score'] == pytest.approx(artifact['r2'])
        assert metrics['mean_absolute_error'] == pytest.approx(artifact['mae'])
        assert metrics['root_mean_squared_error'] == pytest.approx(artifact['rmse'])
        assert metrics['metric_sources']['selected_model']['artifact_path'] == artifact['artifact_path']


class TestModelInfoEndpoint:
    """Test the model-info endpoint."""

    def test_model_info_returns_200(self, client):
        """GET /model-info should return 200."""
        response = client.get('/model-info')
        assert response.status_code == 200

    def test_model_info_returns_json(self, client):
        """Model info should return valid JSON."""
        response = client.get('/model-info')
        data = response.get_json()
        assert data is not None

    def test_api_models_uses_artifact_values(self, client):
        """GET /api/models should surface artifact metrics instead of hardcoded placeholders."""
        response = client.get('/api/models')
        data = response.get_json()
        assert response.status_code == 200
        assert data['success'] is True

        catalog = get_model_catalog()
        original = catalog['original']
        xgboost = catalog['xgboost']
        models = data['models']

        assert models['original']['r2'] == pytest.approx(original['r2'])
        assert models['original']['rmse'] == pytest.approx(original['rmse'])
        assert models['original']['mae'] == pytest.approx(original['mae'])
        assert models['xgboost']['r2'] == pytest.approx(xgboost['r2'])
        assert models['xgboost']['rmse'] == pytest.approx(xgboost['rmse'])
        assert models['xgboost']['mae'] == pytest.approx(xgboost['mae'])


class TestRouteRegistration:
    """Route registration should use the modular backend as the primary app."""

    def test_root_routes_use_modular_blueprints(self, app):
        endpoint_by_rule = {rule.rule: rule.endpoint for rule in app.url_map.iter_rules()}

        assert endpoint_by_rule['/'] == 'pages.home'
        assert endpoint_by_rule['/predict'] == 'prediction.predict'
        assert endpoint_by_rule['/analytics'] == 'analytics.analytics_page'
        assert endpoint_by_rule['/api/status'] == 'pages.api_status'

    def test_v2_routes_remain_available(self, client):
        """The modular /v2 aliases should stay available during migration."""
        assert client.get('/v2/').status_code == 200
        assert client.get('/v2/api/status').status_code == 200

    def test_compatibility_routes_now_point_to_modular_blueprints(self, app):
        endpoint_by_rule = {rule.rule: rule.endpoint for rule in app.url_map.iter_rules()}

        assert endpoint_by_rule['/predict/advanced'] == 'prediction.predict_advanced'
        assert endpoint_by_rule['/predict/xgboost'] == 'prediction.predict_xgboost_route'
        assert endpoint_by_rule['/predict/compare'] == 'prediction.compare_predictions'
        assert endpoint_by_rule['/api/models'] == 'pages.list_models'
        assert endpoint_by_rule['/api/analytics/date-range'] == 'analytics.date_range_analytics'
        assert endpoint_by_rule['/api/send-analytics-email'] == 'export.send_analytics_email'

    def test_no_legacy_compat_blueprint_registered(self, app):
        assert 'legacy_compat' not in app.blueprints
