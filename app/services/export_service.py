"""
Export Service - CSV/JSON/PDF/Excel export logic.
"""

import csv
import io
import json
from datetime import datetime


def predictions_to_csv(predictions: list) -> str:
    """Convert prediction list to CSV string."""
    if not predictions:
        return ""

    output = io.StringIO()
    fieldnames = list(predictions[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in predictions:
        writer.writerow({k: str(v) for k, v in row.items()})
    return output.getvalue()


def predictions_to_json(predictions: list) -> str:
    """Convert prediction list to formatted JSON string."""
    return json.dumps(predictions, indent=2, default=str)


def generate_report_text(metrics: dict, predictions: list) -> str:
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
        bikes = pred.get('predicted_bikes', pred.get('prediction', 'N/A'))
        model = pred.get('model_type', 'unknown')
        ts = pred.get('timestamp_readable', pred.get('timestamp', ''))
        lines.append(f"  {ts} | {model} | {bikes} bikes")

    return "\n".join(lines)
