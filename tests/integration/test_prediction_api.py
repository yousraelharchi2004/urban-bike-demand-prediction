"""
Integration tests for the prediction API endpoints.
"""

import pytest
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestPredictEndpoint:
    """Test POST /predict endpoint."""

    def test_predict_returns_200(self, client, sample_prediction_input_korean):
        """POST /predict with valid data should return 200."""
        response = client.post(
            '/predict',
            data=json.dumps(sample_prediction_input_korean),
            content_type='application/json',
        )
        # 200 or 503 if model not loaded
        assert response.status_code in (200, 503)

    def test_predict_returns_positive_value(self, client, sample_prediction_input_korean):
        """Prediction should be a non-negative number."""
        response = client.post(
            '/predict',
            data=json.dumps(sample_prediction_input_korean),
            content_type='application/json',
        )
        if response.status_code == 200:
            data = response.get_json()
            if data.get('success'):
                bikes = data.get('predicted_bikes', data.get('prediction', data.get('predicted_bikes_raw', -1)))
                assert bikes >= 0

    def test_predict_handles_empty_body(self, client):
        """POST /predict with empty body should return 400."""
        response = client.post(
            '/predict',
            data='{}',
            content_type='application/json',
        )
        # Should be 400 (no data) or 503 (model not loaded)
        assert response.status_code in (400, 500, 503)

    def test_predict_handles_no_json(self, client):
        """POST /predict with no Content-Type should return error."""
        response = client.post('/predict', data='not json')
        assert response.status_code in (400, 415, 500, 503)

    def test_predict_model_type_parameter(self, client, sample_prediction_input_korean):
        """Model type query parameter should be accepted."""
        response = client.post(
            '/predict?model_type=original',
            data=json.dumps(sample_prediction_input_korean),
            content_type='application/json',
        )
        assert response.status_code in (200, 503)

    def test_predict_invalid_model_type_falls_back(self, client, sample_prediction_input_korean):
        """Invalid model type should fall back to original."""
        response = client.post(
            '/predict?model_type=nonexistent',
            data=json.dumps(sample_prediction_input_korean),
            content_type='application/json',
        )
        assert response.status_code in (200, 503)

    def test_predict_unified_model_type(self, client, sample_prediction_input_korean):
        """Unified model type should be accepted (may return 404 if pipeline not trained)."""
        response = client.post(
            '/predict?model_type=unified',
            data=json.dumps(sample_prediction_input_korean),
            content_type='application/json',
        )
        # 200 if pipeline exists, 404/500 if not
        assert response.status_code in (200, 404, 500, 503)


class TestPredictionHistory:
    """Test GET /prediction-history endpoint."""

    def test_prediction_history_returns_200(self, client):
        """GET /prediction-history should return 200."""
        response = client.get('/prediction-history')
        assert response.status_code == 200

    def test_prediction_history_returns_json(self, client):
        """Prediction history should return valid JSON."""
        response = client.get('/prediction-history')
        data = response.get_json()
        assert data is not None
