"""
Integration tests for page and status endpoints.
"""

import pytest
import os
import sys

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
