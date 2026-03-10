"""
Lazy model loading with caching.
Centralizes all model state so blueprints can share it.
"""

from __future__ import annotations

import os
import json
from typing import Any, Optional

import joblib
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model state (module-level singleton)
# ---------------------------------------------------------------------------
_state = {
    'original_model': None,
    'improved_model': None,
    'advanced_model': None,
    'xgboost_model': None,
    'lightgbm_model': None,
    'stacking_model': None,
    'advanced_model_features': None,
    'xgboost_model_features': None,
    'model_loaded': False,
}


def _extract_model(data: Any) -> Any:
    """Extract the actual model from a dict/tuple wrapper."""
    if hasattr(data, 'predict'):
        return data

    if isinstance(data, dict):
        for key in ('model', 'best_model', 'estimator', 'regressor', 'classifier', 'pipeline'):
            if key in data and hasattr(data[key], 'predict'):
                return data[key]
        # Last resort: find first object with .predict
        for v in data.values():
            if hasattr(v, 'predict'):
                return v

    if isinstance(data, (list, tuple)):
        for item in data:
            if hasattr(item, 'predict'):
                return item

    return data


def _try_load(paths: list[str], name: str) -> tuple[Any, Any]:
    """Try loading a model from multiple possible paths."""
    for p in paths:
        if os.path.exists(p):
            try:
                raw = joblib.load(p)
                model = _extract_model(raw)
                if hasattr(model, 'predict'):
                    logger.info(f"Loaded {name} from {p}")
                    return model, raw
            except Exception as e:
                logger.warning(f"Failed to load {name} from {p}: {e}")
    return None, None


def get_state() -> dict:
    """Return the global model state, initializing if needed."""
    if not _state['model_loaded']:
        init_models()
    return _state


def init_models():
    """Load all available models into module state."""
    if _state['model_loaded']:
        return

    base_paths = [
        'models',
        os.path.join(os.path.dirname(__file__), '..', '..', 'models'),
    ]

    def paths_for(filename):
        return [os.path.join(b, filename) for b in base_paths]

    # Original
    model, _ = _try_load(paths_for('random_forest_model.pkl'), 'original')
    _state['original_model'] = model

    # Improved
    model, _ = _try_load(paths_for('improved_random_forest.pkl'), 'improved')
    _state['improved_model'] = model

    # Advanced
    model, raw = _try_load(paths_for('advanced_random_forest.pkl'), 'advanced')
    _state['advanced_model'] = model
    if model and hasattr(model, 'feature_names_in_'):
        _state['advanced_model_features'] = list(model.feature_names_in_)
    elif isinstance(raw, dict) and 'feature_names' in raw:
        _state['advanced_model_features'] = raw['feature_names']
    else:
        # Try loading from JSON
        for b in base_paths:
            info_path = os.path.join(b, 'advanced_model_info.json')
            if os.path.exists(info_path):
                try:
                    with open(info_path) as f:
                        _state['advanced_model_features'] = json.load(f).get('feature_names', [])
                except (json.JSONDecodeError, KeyError):
                    pass

    # XGBoost
    model, raw = _try_load(paths_for('xgboost_model.pkl'), 'xgboost')
    _state['xgboost_model'] = model
    if isinstance(raw, dict) and 'feature_names' in raw:
        _state['xgboost_model_features'] = raw['feature_names']

    # LightGBM
    model, _ = _try_load(paths_for('lightgbm_model.pkl'), 'lightgbm')
    _state['lightgbm_model'] = model

    # Stacking Ensemble
    model, _ = _try_load(paths_for('stacking_model.pkl'), 'stacking')
    _state['stacking_model'] = model

    _state['model_loaded'] = True
    loaded = sum(1 for k in ('original_model', 'improved_model', 'advanced_model',
                              'xgboost_model', 'lightgbm_model', 'stacking_model')
                 if _state[k] is not None)
    logger.info(f"Models initialized: {loaded}/6 loaded")
