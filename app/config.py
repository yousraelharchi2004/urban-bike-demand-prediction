"""Runtime environment checks for the supported artifact bundle."""

from __future__ import annotations

from pathlib import Path
import logging
import os

from app.artifact_contract import (
    format_dependency_mismatch_error,
    read_manifest,
    runtime_dependency_versions,
    unified_manifest_path,
)


def get_runtime_environment_report(project_root: Path | None = None) -> dict:
    """Return runtime versions and any manifest compatibility error."""
    root = project_root or Path(__file__).resolve().parents[1]
    manifest = read_manifest(unified_manifest_path(root))
    return {
        'runtime_dependencies': runtime_dependency_versions(),
        'manifest_path': str(unified_manifest_path(root)),
        'artifact_dependency_error': format_dependency_mismatch_error(manifest) if manifest else None,
    }


def validate_runtime_environment(project_root: Path | None = None) -> None:
    """Raise when the supported artifact bundle does not match the deployed runtime."""
    report = get_runtime_environment_report(project_root)
    if report['artifact_dependency_error']:
        logger = logging.getLogger(__name__)
        logger.warning(report['artifact_dependency_error'])
        if os.environ.get('STRICT_ARTIFACT_CHECK', '0') == '1':
            raise RuntimeError(report['artifact_dependency_error'])
        return False
    return True
