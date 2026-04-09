"""Regression tests for the non-interactive runtime entrypoint."""

from pathlib import Path


def test_run_py_has_no_desktop_side_effects():
    source = Path('run.py').read_text(encoding='utf-8')

    assert 'import webbrowser' not in source
    assert 'open_browser(' not in source
    assert 'input(' not in source
    assert 'Continue anyway?' not in source
