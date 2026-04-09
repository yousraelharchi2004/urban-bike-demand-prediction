from __future__ import annotations

from importlib import metadata
from pathlib import Path

PACKAGE_LOCK = {
    'scikit-learn',
    'pandas',
    'numpy',
    'joblib',
    'xgboost',
    'lightgbm',
}


def _read_locked_versions() -> dict[str, str]:
    requirements_path = Path(__file__).resolve().parents[1] / 'requirements.txt'
    if not requirements_path.exists():
        return {}

    versions: dict[str, str] = {}
    for line in requirements_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '==' not in line:
            continue
        name, version = (part.strip() for part in line.split('==', 1))
        if name in PACKAGE_LOCK:
            versions[name] = version
    return versions


def test_environment_matches_requirements() -> None:
    """Ensure the runtime packages used for tests match the pinned versions."""
    locked_versions = _read_locked_versions()
    mismatches: list[str] = []

    for package, expected_version in locked_versions.items():
        try:
            runtime_version = metadata.version(package)
        except metadata.PackageNotFoundError:
            mismatches.append(f'{package} is not installed (expected {expected_version})')
            continue
        if runtime_version != expected_version:
            mismatches.append(
                f'{package} version mismatch: requirements=={expected_version} but runtime=={runtime_version}'
            )

    if mismatches:
        diagnostic = '\n'.join(f'- {line}' for line in mismatches)
        raise AssertionError(
            'Test environment drift detected. Please recreate the environment with:\n'
            '  pip install -r requirements.txt\n'
            f'{diagnostic}'
        )
