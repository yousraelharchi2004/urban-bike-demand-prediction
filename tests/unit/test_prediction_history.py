"""
Unit tests for prediction history schema normalization and migration.
"""

import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestPredictionHistorySchema:
    """Prediction history should converge to one canonical stored schema."""

    def test_get_prediction_history_migrates_legacy_entries(self, monkeypatch):
        from app import utils

        history_path = Path('data') / f'test_predictions_history_{uuid.uuid4().hex}.json'
        try:
            history_path.write_text(json.dumps([
                {
                    "timestamp": "2026-04-04T14:17:55.694613",
                    "input": {"Hour": 14},
                    "prediction": 863.8469,
                    "rounded": 864,
                    "confidence": 12,
                    "model_type": "unified",
                    "success": True,
                }
            ]), encoding='utf-8')
            monkeypatch.setattr(utils, 'PREDICTION_HISTORY_FILE', str(history_path))

            history = utils.get_prediction_history(limit=None)

            assert len(history) == 1
            entry = history[0]
            assert entry['prediction_id'] == 1
            assert entry['input_data'] == {"Hour": 14}
            assert entry['predicted_bikes_raw'] == 863.8469
            assert entry['predicted_bikes'] == 864
            assert entry['schema_version'] == utils.PREDICTION_HISTORY_SCHEMA_VERSION
            assert 'input' not in entry
            assert 'prediction' not in entry
            assert 'rounded' not in entry

            persisted = json.loads(history_path.read_text(encoding='utf-8'))
            assert persisted == history
        finally:
            history_path.unlink(missing_ok=True)

    def test_save_prediction_to_history_writes_canonical_entry(self, monkeypatch):
        from app import utils

        history_path = Path('data') / f'test_predictions_history_{uuid.uuid4().hex}.json'
        try:
            monkeypatch.setattr(utils, 'PREDICTION_HISTORY_FILE', str(history_path))

            utils.save_prediction_to_history({"Hour": 8}, 123.4, model_type='original')

            persisted = json.loads(history_path.read_text(encoding='utf-8'))
            assert len(persisted) == 1
            entry = persisted[0]
            assert entry['prediction_id'] == 1
            assert entry['input_data'] == {"Hour": 8}
            assert entry['predicted_bikes_raw'] == 123.4
            assert entry['predicted_bikes'] == 123
            assert entry['model_type'] == 'original'
            assert entry['success'] is True
            assert entry['schema_version'] == utils.PREDICTION_HISTORY_SCHEMA_VERSION
            assert set(entry) == {
                'prediction_id',
                'timestamp',
                'timestamp_readable',
                'input_data',
                'predicted_bikes_raw',
                'predicted_bikes',
                'confidence',
                'model_type',
                'success',
                'schema_version',
            }
        finally:
            history_path.unlink(missing_ok=True)

    def test_prediction_ids_are_sequenced_per_day(self, monkeypatch):
        from app import utils

        history_path = Path('data') / f'test_predictions_history_{uuid.uuid4().hex}.json'
        try:
            history_path.write_text(json.dumps([
                {
                    "timestamp": "2026-04-06T09:00:00",
                    "timestamp_readable": "2026-04-06 09:00:00",
                    "input_data": {"Hour": 9},
                    "predicted_bikes_raw": 100.0,
                    "predicted_bikes": 100,
                    "confidence": 0,
                    "model_type": "original",
                    "success": True,
                    "schema_version": utils.PREDICTION_HISTORY_SCHEMA_VERSION,
                },
                {
                    "timestamp": "2026-04-06T10:00:00",
                    "timestamp_readable": "2026-04-06 10:00:00",
                    "input_data": {"Hour": 10},
                    "predicted_bikes_raw": 200.0,
                    "predicted_bikes": 200,
                    "confidence": 0,
                    "model_type": "original",
                    "success": True,
                    "schema_version": utils.PREDICTION_HISTORY_SCHEMA_VERSION,
                    "prediction_id": 99,
                },
                {
                    "timestamp": "2026-04-07T08:00:00",
                    "timestamp_readable": "2026-04-07 08:00:00",
                    "input_data": {"Hour": 8},
                    "predicted_bikes_raw": 300.0,
                    "predicted_bikes": 300,
                    "confidence": 0,
                    "model_type": "original",
                    "success": True,
                    "schema_version": utils.PREDICTION_HISTORY_SCHEMA_VERSION,
                    "prediction_id": 101,
                }
            ]), encoding='utf-8')
            monkeypatch.setattr(utils, 'PREDICTION_HISTORY_FILE', str(history_path))

            history = utils.get_prediction_history(limit=None)

            assert [entry['prediction_id'] for entry in history] == [1, 2, 1]
        finally:
            history_path.unlink(missing_ok=True)
