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

    @staticmethod
    def _assert_success_payload(data, expected_model_type):
        assert data['success'] is True
        assert data['model_type'] == expected_model_type
        assert isinstance(data['predicted_bikes'], int)
        assert data['predicted_bikes'] >= 0
        assert isinstance(data['predicted_bikes_raw'], (int, float))
        assert data['predicted_bikes_raw'] >= 0
        assert isinstance(data['prediction_id'], int)
        assert data['prediction_id'] >= 1
        assert isinstance(data['timestamp'], str)
        assert 'error' not in data

    def test_predict_returns_200(self, client, sample_prediction_input_korean):
        """POST /predict with valid data should return 200."""
        response = client.post(
            '/predict',
            data=json.dumps(sample_prediction_input_korean),
            content_type='application/json',
        )
        assert response.status_code == 200
        self._assert_success_payload(response.get_json(), 'original')

    def test_predict_returns_positive_value(self, client, sample_prediction_input_korean):
        """Prediction should be a non-negative number."""
        response = client.post(
            '/predict',
            data=json.dumps(sample_prediction_input_korean),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = response.get_json()
        self._assert_success_payload(data, 'original')
        assert data['predicted_bikes'] == round(data['predicted_bikes_raw'])

    def test_predict_handles_empty_body(self, client):
        """POST /predict with empty body should return 400."""
        response = client.post(
            '/predict',
            data='{}',
            content_type='application/json',
        )
        assert response.status_code == 400
        assert response.get_json() == {
            'success': False,
            'error': 'No data provided',
        }

    def test_predict_handles_no_json(self, client):
        """POST /predict with no Content-Type should return error."""
        response = client.post('/predict', data='not json')
        assert response.status_code == 415
        assert response.get_json() == {
            'success': False,
            'error': 'Content-Type must be application/json',
        }

    def test_predict_model_type_parameter(self, client, sample_prediction_input_korean):
        """Model type query parameter should be accepted."""
        response = client.post(
            '/predict?model_type=original',
            data=json.dumps(sample_prediction_input_korean),
            content_type='application/json',
        )
        assert response.status_code == 200
        self._assert_success_payload(response.get_json(), 'original')

    def test_predict_invalid_model_type_falls_back(self, client, sample_prediction_input_korean):
        """Invalid model type should fall back to original."""
        response = client.post(
            '/predict?model_type=nonexistent',
            data=json.dumps(sample_prediction_input_korean),
            content_type='application/json',
        )
        assert response.status_code == 200
        self._assert_success_payload(response.get_json(), 'original')

    def test_predict_unified_model_type(self, client, sample_prediction_input_korean):
        """Unified model type should return the unified model payload."""
        response = client.post(
            '/predict?model_type=unified',
            data=json.dumps(sample_prediction_input_korean),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = response.get_json()
        self._assert_success_payload(data, 'unified')
        assert data['model_name'] == 'stacking_ensemble'


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
        assert data['success'] is True
        assert isinstance(data['predictions'], list)
        assert isinstance(data['count'], int)


class TestSpecificModelTypes:
    """Concrete regression checks for each model route contract."""

    def test_all_available_model_types(self, client, sample_prediction_input_korean):
        expected_status = {
            'original': 200,
            'improved': 200,
            'advanced': 200,
            'xgboost': 200,
            'lightgbm': 200,
            'stacking': 200,
            'unified': 200,
        }

        payload = dict(sample_prediction_input_korean)
        payload['day_of_week'] = 3

        for model_type, status_code in expected_status.items():
            response = client.post(
                f'/predict?model_type={model_type}',
                data=json.dumps(payload),
                content_type='application/json',
            )
            assert response.status_code == status_code, f'{model_type}: {response.get_json()}'
            data = response.get_json()
            assert data['success'] is True
            assert data['model_type'] == model_type
            assert data['predicted_bikes'] == round(data['predicted_bikes_raw'])

    def test_compatibility_prediction_routes_use_modular_handlers(self, client, sample_prediction_input_korean):
        payload = dict(sample_prediction_input_korean)
        payload['day_of_week'] = 3

        advanced = client.post('/predict/advanced', data=json.dumps(payload), content_type='application/json')
        xgboost = client.post('/predict/xgboost', data=json.dumps(payload), content_type='application/json')

        assert advanced.status_code == 200
        assert xgboost.status_code == 200
        assert advanced.get_json()['model_type'] == 'advanced'
        assert xgboost.get_json()['model_type'] == 'xgboost'

        compare = client.post('/predict/compare', data=json.dumps(payload), content_type='application/json')
        assert compare.status_code == 200
        data = compare.get_json()
        assert data['success'] is True
        assert 'results' in data
        assert data['models_compared'] == 7
