"""
Unit tests for the supported unified artifact contract.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestUnifiedArtifactContract:
    """The unified pipeline should expose one canonical manifest."""

    def test_manifest_exists_and_matches_bundle(self):
        from app.artifact_contract import runtime_dependency_versions, unified_manifest_path

        manifest_path = unified_manifest_path(Path.cwd())
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        assert manifest['contract_version'] == 1
        assert manifest['pipeline'] == 'unified'
        assert manifest['default_model'] == 'stacking_ensemble'
        assert manifest['runtime_dependencies'] == runtime_dependency_versions()
        assert set(manifest['models']) >= {
            'random_forest',
            'gradient_boosting',
            'xgboost',
            'lightgbm',
            'stacking_ensemble',
        }

        base_dir = manifest_path.parent
        assert (base_dir / manifest['feature_names_path']).exists()
        assert (base_dir / manifest['scaler_path']).exists()
        assert (base_dir / manifest['metrics_path']).exists()
        for model in manifest['models'].values():
            assert (base_dir / model['artifact_path']).exists()

    def test_supported_artifacts_do_not_emit_version_mismatch_warnings(self):
        from sklearn.exceptions import InconsistentVersionWarning

        artifact_paths = [
            Path('models/random_forest_model.pkl'),
            Path('models/unified_pipeline/stacking_ensemble.pkl'),
        ]

        for artifact_path in artifact_paths:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                joblib.load(artifact_path)
            mismatches = [w for w in caught if issubclass(w.category, InconsistentVersionWarning)]
            assert not mismatches, f'{artifact_path} emitted {mismatches}'
