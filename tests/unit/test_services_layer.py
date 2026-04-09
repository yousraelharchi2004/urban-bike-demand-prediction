"""Regression checks that business logic stays in app.services."""

from pathlib import Path

from app.services.export_service import history_rows, predictions_to_csv, predictions_to_json


def test_prediction_blueprint_delegates_to_service_module():
    source = Path('app/blueprints/prediction.py').read_text(encoding='utf-8')

    assert 'from app.services import prediction_service' in source
    assert 'prediction_service.predict(' in source
    assert 'prediction_service.compare_predictions(' in source
    assert 'prediction_service.load_history(' in source


def test_export_blueprint_uses_export_service_helpers():
    source = Path('app/blueprints/export.py').read_text(encoding='utf-8')

    assert 'from app.services.export_service import' in source
    assert 'history_rows(' in source
    assert 'predictions_to_csv(' in source
    assert 'predictions_to_json(' in source
    assert 'filtered_history(' in source


def test_export_service_serializes_prediction_rows():
    rows = history_rows([
        {
            'timestamp': '2026-04-06T12:00:00',
            'timestamp_readable': '2026-04-06 12:00:00',
            'model_type': 'original',
            'predicted_bikes': 321,
            'predicted_bikes_raw': 321.4,
            'input_data': {
                'Hour': 12,
                'Temperature(Â°C)': 21.5,
                'Seasons_Summer': 1,
            },
        }
    ])

    assert rows[0]['prediction'] == 321
    assert rows[0]['temperature'] == 21.5
    assert rows[0]['season'] == 'Summer'
    assert 'predicted_bikes_raw' in predictions_to_csv(rows)
    assert '"model_type": "original"' in predictions_to_json(rows)
