"""Deprecated wrapper for the unified training pipeline."""

from training.legacy.deprecated_training_entry import run_deprecated_entry


if __name__ == '__main__':
    run_deprecated_entry('training/legacy/train_model_fixed.py')
