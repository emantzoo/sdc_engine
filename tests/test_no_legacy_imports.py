"""Fail the build if anyone re-introduces the legacy select_method_by_features import."""
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
FORBIDDEN = [
    "from sdc_engine.sdc.select_method import select_method_by_features",
    "from .select_method import select_method_by_features",
]


def test_no_legacy_select_method_imports():
    violations = []
    for py_file in REPO_ROOT.rglob("*.py"):
        # Skip the stub file itself and archived code
        if "_archive" in str(py_file) or py_file.name in ("select_method.py", "test_no_legacy_imports.py"):
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN:
            if pattern in text:
                violations.append(f"{py_file}: {pattern}")
    assert not violations, (
        "Legacy select_method_by_features import found. "
        "Use sdc_engine.sdc.selection instead:\n" + "\n".join(violations)
    )


def test_legacy_stub_raises():
    """Sanity: the stub function raises ImportError."""
    from sdc_engine.sdc.select_method import select_method_by_features
    with pytest.raises(ImportError, match="moved to sdc_engine.sdc.selection"):
        select_method_by_features(None, None)


def test_canonical_export_works():
    """The package-level export should point to the canonical function."""
    from sdc_engine.sdc import select_method_by_features
    # Should be the real function from selection.rules, not the stub
    assert select_method_by_features.__module__ == "sdc_engine.sdc.selection.rules"
