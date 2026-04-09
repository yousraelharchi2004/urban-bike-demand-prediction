#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/app"
MANIFEST="${APP_DIR}/models/unified_pipeline/artifact_manifest.json"

echo "[start] verifying required artifacts"
if [ ! -f "$MANIFEST" ]; then
    echo "[start] missing manifest: $MANIFEST" >&2
    exit 1
fi

echo "[start] manifest present, starting gunicorn"
exec gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 run:app
