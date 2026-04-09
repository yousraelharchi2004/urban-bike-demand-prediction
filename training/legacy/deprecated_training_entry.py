"""
Helpers for deprecated training entrypoints.
"""

from __future__ import annotations


def run_deprecated_entry(script_name: str) -> None:
    """Route deprecated training scripts to the unified pipeline."""
    print(f"[DEPRECATED] {script_name} is no longer a supported training pipeline.")
    print("[DEPRECATED] Use python train_unified_pipeline.py")

    from train_unified_pipeline import main
    main()
