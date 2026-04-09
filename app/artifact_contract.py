"""
Canonical artifact contract for the officially supported training pipeline.
"""

from __future__ import annotations

import json
import importlib.metadata
from datetime import datetime
from pathlib import Path


ARTIFACT_CONTRACT_VERSION = 1
UNIFIED_PIPELINE_DIRNAME = 'unified_pipeline'
UNIFIED_MANIFEST_FILENAME = 'artifact_manifest.json'


def unified_pipeline_dir(project_root: Path) -> Path:
    return project_root / 'models' / UNIFIED_PIPELINE_DIRNAME


def unified_manifest_path(project_root: Path) -> Path:
    return unified_pipeline_dir(project_root) / UNIFIED_MANIFEST_FILENAME


def build_unified_manifest(*, config: dict, feature_names_path: str, scaler_path: str,
                           lag_medians_path: str, demand_cache_path: str,
                           metrics_path: str, model_filenames: dict[str, str]) -> dict:
    """Build the canonical manifest for unified pipeline artifacts."""
    best_model = config.get('best_model', 'stacking_ensemble')
    return {
        'contract_version': ARTIFACT_CONTRACT_VERSION,
        'pipeline': 'unified',
        'created_at': datetime.now().isoformat(),
        'default_model': best_model,
        'feature_names_path': feature_names_path,
        'scaler_path': scaler_path,
        'lag_medians_path': lag_medians_path,
        'demand_cache_path': demand_cache_path,
        'config_path': 'pipeline_config.json',
        'metrics_path': metrics_path,
        'runtime_dependencies': runtime_dependency_versions(),
        'models': {
            model_name: {
                'artifact_path': artifact_path,
                'metrics_key': model_name,
            }
            for model_name, artifact_path in model_filenames.items()
        },
    }


def read_manifest(path: Path) -> dict:
    """Load a manifest file, returning {} if unreadable."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def runtime_dependency_versions() -> dict[str, str]:
    """Return the deployed package versions relevant to artifact compatibility."""
    packages = (
        'scikit-learn',
        'pandas',
        'numpy',
        'joblib',
        'xgboost',
        'lightgbm',
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def manifest_dependency_mismatches(manifest: dict) -> dict[str, dict[str, str]]:
    """Return manifest/runtime dependency mismatches keyed by package name."""
    expected = manifest.get('runtime_dependencies') or {}
    current = runtime_dependency_versions()
    mismatches: dict[str, dict[str, str]] = {}
    for package, expected_version in expected.items():
        current_version = current.get(package)
        if current_version and current_version != expected_version:
            mismatches[package] = {
                'expected': expected_version,
                'current': current_version,
            }
    return mismatches


def format_dependency_mismatch_error(manifest: dict) -> str | None:
    """Return a readable compatibility error for artifact/runtime mismatches."""
    mismatches = manifest_dependency_mismatches(manifest)
    if not mismatches:
        return None

    parts = [
        f"{package} artifact={detail['expected']} runtime={detail['current']}"
        for package, detail in sorted(mismatches.items())
    ]
    return (
        'Artifact dependency mismatch detected. '
        'Rebuild the supported artifacts with the deployed environment. '
        + '; '.join(parts)
    )
