"""
Prediction Service - Core prediction logic, history, confidence calculation.
"""

import json
import os
import math
import numpy as np
from datetime import datetime

PREDICTION_HISTORY_FILE = 'data/predictions_history.json'


def calculate_confidence(predicted_bikes: float, input_data: dict) -> float:
    """Calculate a confidence score (0-100) for a prediction."""
    confidence = 85.0  # base

    # Penalize extreme predictions
    if predicted_bikes > 3000 or predicted_bikes < 0:
        confidence -= 15
    elif predicted_bikes > 2000:
        confidence -= 5

    # Penalize extreme weather
    temp = float(input_data.get('Temperature(°C)', input_data.get('temperature', 20)))
    if temp < -10 or temp > 40:
        confidence -= 10
    elif temp < 0 or temp > 35:
        confidence -= 5

    rain = float(input_data.get('Rainfall(mm)', input_data.get('rainfall', 0)))
    if rain > 10:
        confidence -= 10

    return max(0, min(100, round(confidence, 1)))


def save_prediction(input_data: dict, prediction: float, model_type: str = 'original'):
    """Save a prediction to the history file."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'timestamp_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'predicted_bikes': round(prediction),
        'predicted_bikes_raw': prediction,
        'model_type': model_type,
        'confidence': calculate_confidence(prediction, input_data),
        'input_data': input_data,
    }

    history = load_history(limit=None)
    history.append(entry)

    # Keep last 1000
    if len(history) > 1000:
        history = history[-1000:]

    os.makedirs(os.path.dirname(PREDICTION_HISTORY_FILE), exist_ok=True)
    with open(PREDICTION_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)

    return entry


def load_history(limit: int = 50) -> list:
    """Load prediction history from file."""
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        return []
    try:
        with open(PREDICTION_HISTORY_FILE) as f:
            data = json.load(f)
        if limit:
            return data[-limit:]
        return data
    except (json.JSONDecodeError, FileNotFoundError):
        return []
