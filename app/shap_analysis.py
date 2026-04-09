"""On-demand SHAP analysis generation and caching for supported models."""

from __future__ import annotations

import json
import io
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from app.models.loader import get_state
from app.model_metrics import get_model_catalog
from app.utils import (
    preprocess_original_features,
    preprocess_with_advanced_features,
    preprocess_with_engineered_features,
)

try:
    import shap
except ModuleNotFoundError:
    shap = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "seoul_bike.csv"

TREE_SHAP_SUPPORTED_MODELS = {"original", "improved", "advanced", "xgboost", "lightgbm", "stacking"}
SHAP_ARTIFACT_PATHS = {
    "original": MODELS_DIR / "shap_analysis_original.json",
    "improved": MODELS_DIR / "shap_analysis_improved.json",
    "advanced": MODELS_DIR / "shap_analysis_advanced.json",
    "xgboost": MODELS_DIR / "shap_analysis_xgboost.json",
    "lightgbm": MODELS_DIR / "shap_analysis.json",
    "stacking": MODELS_DIR / "shap_analysis_stacking.json",
}
STATE_KEYS = {
    "original": "original_model",
    "improved": "improved_model",
    "advanced": "advanced_model",
    "xgboost": "xgboost_model",
    "lightgbm": "lightgbm_model",
    "stacking": "stacking_model",
}


def _source_label(model_key: str) -> str:
    return f"{get_model_catalog().get(model_key, {}).get('name', model_key)} SHAP Analysis"


def _artifact_path(model_key: str) -> Path | None:
    return SHAP_ARTIFACT_PATHS.get(model_key)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_raw_dataset() -> pd.DataFrame:
    return pd.read_csv(RAW_DATASET_PATH, encoding="latin1")


def _row_to_input(row: pd.Series) -> dict:
    date_value = pd.to_datetime(row["Date"], format="%d/%m/%Y", errors="coerce")
    return {
        "Hour": float(row["Hour"]),
        "Temperature(°C)": float(row["Temperature(°C)"]),
        "Humidity(%)": float(row["Humidity(%)"]),
        "Wind speed (m/s)": float(row["Wind speed (m/s)"]),
        "Visibility (10m)": float(row["Visibility (10m)"]),
        "Dew point temperature(°C)": float(row["Dew point temperature(°C)"]),
        "Solar Radiation (MJ/m2)": float(row["Solar Radiation (MJ/m2)"]),
        "Rainfall(mm)": float(row["Rainfall(mm)"]),
        "Snowfall (cm)": float(row["Snowfall (cm)"]),
        "Seasons": row["Seasons"],
        "Holiday": row["Holiday"],
        "Functioning Day": row["Functioning Day"],
        "day_of_week": int(date_value.dayofweek if not pd.isna(date_value) else 0),
    }


def _prepare_feature_frame(model_key: str, model, sample_size: int) -> pd.DataFrame:
    raw_dataset = _load_raw_dataset()
    raw = raw_dataset.sample(n=min(sample_size, len(raw_dataset)), random_state=42).reset_index(drop=True)
    rows = []
    for _, row in raw.iterrows():
        input_data = _row_to_input(row)
        with redirect_stdout(io.StringIO()):
            if model_key == "original":
                processed = preprocess_original_features(input_data, model=model)
            elif model_key == "improved":
                processed = preprocess_with_engineered_features(input_data, model=model)
            else:
                processed = preprocess_with_advanced_features(input_data, model=model)
        rows.append(processed.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _compute_tree_shap(model, features: pd.DataFrame) -> tuple[np.ndarray, float | None]:
    if shap is None:
        raise ModuleNotFoundError("shap is not installed")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_array = np.asarray(shap_values, dtype=float)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.asarray(expected_value).reshape(-1)[0])
    elif expected_value is not None:
        expected_value = float(expected_value)
    return shap_array, expected_value


def _compute_model_agnostic_shap(model, features: pd.DataFrame) -> tuple[np.ndarray, float | None]:
    if shap is None:
        raise ModuleNotFoundError("shap is not installed")

    feature_values = features.to_numpy(dtype=float)
    background_size = min(12, len(features))
    explain_size = min(16, len(features))
    background = feature_values[:background_size]
    explain_frame = feature_values[:explain_size]

    def predict_fn(values):
        return np.asarray(model.predict(np.asarray(values, dtype=float)), dtype=float)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain_frame, nsamples=50)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_array = np.asarray(shap_values, dtype=float)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.asarray(expected_value).reshape(-1)[0])
    elif expected_value is not None:
        expected_value = float(expected_value)
    return shap_array, expected_value


def _build_payload(model_key: str, features: pd.DataFrame, shap_values: np.ndarray, expected_value: float | None) -> dict:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_signed = np.mean(shap_values, axis=0)
    order = np.argsort(mean_abs)[::-1]

    global_importance = [
        {
            "feature": str(features.columns[idx]),
            "mean_abs_shap": float(mean_abs[idx]),
        }
        for idx in order[:15]
    ]
    direction = [
        {
            "feature": str(features.columns[idx]),
            "mean_shap": float(mean_signed[idx]),
        }
        for idx in order[:15]
    ]

    return {
        "model_key": model_key,
        "source": _source_label(model_key),
        "global_importance": global_importance,
        "direction": direction,
        "expected_value": expected_value,
        "sample_size": int(len(features)),
    }


def get_or_generate_shap_analysis(model_key: str, force: bool = False) -> dict:
    artifact_path = _artifact_path(model_key)
    if model_key not in TREE_SHAP_SUPPORTED_MODELS or artifact_path is None:
        return {
            "success": True,
            "features": [],
            "source": _source_label(model_key),
            "requested_model": model_key,
            "available": False,
        }

    if not force:
        cached = _read_json(artifact_path)
        if cached.get("global_importance"):
            return {
                "success": True,
                "features": [
                    {
                        "feature": item["feature"],
                        "importance": round(item["mean_abs_shap"], 2),
                        "direction": "positive" if next((d["mean_shap"] for d in cached.get("direction", []) if d["feature"] == item["feature"]), 0) > 0 else "negative",
                        "mean_shap": round(next((d["mean_shap"] for d in cached.get("direction", []) if d["feature"] == item["feature"]), 0), 2),
                    }
                    for item in cached.get("global_importance", [])[:15]
                ],
                "expected_value": cached.get("expected_value"),
                "source": cached.get("source", _source_label(model_key)),
                "requested_model": model_key,
                "available": True,
            }

    if shap is None:
        return {
            "success": True,
            "features": [],
            "source": _source_label(model_key),
            "requested_model": model_key,
            "available": False,
            "error": "shap package is not installed",
        }

    state = get_state()
    model = state.get(STATE_KEYS[model_key])
    if model is None:
        return {
            "success": True,
            "features": [],
            "source": _source_label(model_key),
            "requested_model": model_key,
            "available": False,
        }

    features = _prepare_feature_frame(model_key, model, sample_size=32)
    if model_key == "stacking":
        shap_values, expected_value = _compute_model_agnostic_shap(model, features)
        features = features.iloc[: len(shap_values)].reset_index(drop=True)
    else:
        shap_values, expected_value = _compute_tree_shap(model, features)
    payload = _build_payload(model_key, features, shap_values, expected_value)
    _write_json(artifact_path, payload)

    return {
        "success": True,
        "features": [
            {
                "feature": item["feature"],
                "importance": round(item["mean_abs_shap"], 2),
                "direction": "positive" if next((d["mean_shap"] for d in payload["direction"] if d["feature"] == item["feature"]), 0) > 0 else "negative",
                "mean_shap": round(next((d["mean_shap"] for d in payload["direction"] if d["feature"] == item["feature"]), 0), 2),
            }
            for item in payload["global_importance"][:15]
        ],
        "expected_value": payload.get("expected_value"),
        "source": payload.get("source", _source_label(model_key)),
        "requested_model": model_key,
        "available": True,
    }
