"""Regression tests for the non-interactive runtime entrypoint."""

from pathlib import Path


def test_run_py_has_no_blocking_interactive_prompts():
    source = Path('run.py').read_text(encoding='utf-8')

    assert 'input(' not in source
    assert 'Continue anyway?' not in source
