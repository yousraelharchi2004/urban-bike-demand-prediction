"""Export business logic and serialization helpers."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta

from app.utils import extract_prediction_from_history, get_prediction_history


def history_rows(history: list[dict]) -> list[dict]:
    """Normalize prediction history into export-friendly rows."""
    rows = []
    for item in history:
        inp = item.get('input_data', {})
        rounded, raw = extract_prediction_from_history(item)
        rows.append({
            'timestamp': item.get('timestamp_readable', item.get('timestamp')),
            'model_type': item.get('model_type', 'unknown'),
            'prediction': rounded,
            'predicted_bikes': rounded,
            'predicted_bikes_raw': raw,
            'hour': inp.get('Hour', inp.get('hour', '')),
            'temperature': inp.get('Temperature(Â°C)', inp.get('Temperature(ìºœ)', inp.get('temperature', ''))),
            'season': (
                'Summer' if inp.get('Seasons_Summer') == 1 else
                'Winter' if inp.get('Seasons_Winter') == 1 else
                'Spring' if inp.get('Seasons_Spring') == 1 else
                'Autumn'
            ),
        })
    return rows


def filtered_history(filter_name: str = 'all') -> list[dict]:
    """Return prediction history filtered to a time window."""
    history = get_prediction_history(limit=None)
    if filter_name == 'all':
        return history

    cutoff_map = {
        'today': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        'week': datetime.now() - timedelta(days=7),
        'month': datetime.now() - timedelta(days=30),
        'quarter': datetime.now() - timedelta(days=90),
        'year': datetime.now() - timedelta(days=365),
    }
    cutoff = cutoff_map.get(filter_name)
    if cutoff is None:
        return history

    filtered = []
    for item in history:
        ts = item.get('timestamp', item.get('timestamp_readable', ''))
        try:
            if 'T' in str(ts):
                dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00').split('+')[0])
            else:
                dt = datetime.strptime(str(ts)[:19], '%Y-%m-%d %H:%M:%S')
        except (TypeError, ValueError):
            continue
        if dt >= cutoff:
            filtered.append(item)
    return filtered


def predictions_to_csv(predictions: list[dict]) -> str:
    """Convert prediction rows to CSV."""
    output = io.StringIO()
    fieldnames = list(predictions[0].keys()) if predictions else ['timestamp', 'model_type', 'prediction']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in predictions:
        writer.writerow({k: str(v) for k, v in row.items()})
    return output.getvalue()


def predictions_to_json(predictions: list[dict]) -> str:
    """Convert prediction rows to formatted JSON."""
    return json.dumps(predictions, indent=2, default=str)


def generate_report_text(metrics: dict, predictions: list[dict]) -> str:
    """Generate a text report summarizing model performance and predictions."""
    lines = [
        "BIKE DEMAND PREDICTION - MODEL REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "MODEL PERFORMANCE",
        "-" * 30,
    ]

    for key, value in metrics.items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append(f"PREDICTION HISTORY ({len(predictions)} entries)")
    lines.append("-" * 30)

    for pred in predictions[-10:]:
        bikes = extract_prediction_from_history(pred)[0]
        model = pred.get('model_type', 'unknown')
        ts = pred.get('timestamp_readable', pred.get('timestamp', ''))
        lines.append(f"  {ts} | {model} | {bikes} bikes")

    return "\n".join(lines)
