"""Systemic test: every method name used by any rule must appear in
METRIC_ALLOWED_METHODS for at least one metric.

Without this, new methods silently fail the _all_allowed() / _is_allowed()
gate in select_method_suite, and the rule appears to work in unit tests
(which don't check metric compatibility) but is dead in production.

History: GENERALIZE and GENERALIZE_FIRST were missing from
METRIC_ALLOWED_METHODS from inception until Fix 0 (2026-04-20),
silently blocking GEO1, RC4, and QR0 for all metrics.
"""
import ast
import re
from pathlib import Path

import pytest

from sdc_engine.sdc.config import METRIC_ALLOWED_METHODS


# All methods that appear in any metric's allowed list
_ALL_ALLOWED = set()
for methods in METRIC_ALLOWED_METHODS.values():
    _ALL_ALLOWED.update(methods)

# Source files that define rules and pipelines
_SELECTION_DIR = Path(__file__).resolve().parents[1] / "sdc_engine" / "sdc" / "selection"
_RULE_FILES = [_SELECTION_DIR / "rules.py", _SELECTION_DIR / "pipelines.py"]


def _extract_method_names(filepath: Path) -> set:
    """Extract all method name strings from 'method': '...' and 'pipeline': [...]
    patterns in a Python source file using AST."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = set()

    for node in ast.walk(tree):
        # Match dict entries like 'method': 'kANON'
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == 'method'
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)):
                    methods.add(value.value)
                # Match 'pipeline': ['NOISE', 'PRAM']
                if (isinstance(key, ast.Constant) and key.value == 'pipeline'
                        and isinstance(value, ast.List)):
                    for elt in value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            methods.add(elt.value)

    return methods


def _all_rule_methods() -> set:
    """Collect every method name referenced in rules.py and pipelines.py."""
    all_methods = set()
    for fp in _RULE_FILES:
        if fp.exists():
            all_methods.update(_extract_method_names(fp))
    return all_methods


class TestMethodMetricCoverage:
    """Every method used by a rule must be registered in METRIC_ALLOWED_METHODS."""

    def test_all_rule_methods_are_in_allowed_list(self):
        """No method should be silently rejected by _is_allowed() for all metrics."""
        rule_methods = _all_rule_methods()
        assert rule_methods, "Failed to extract any method names from rule files"

        missing = rule_methods - _ALL_ALLOWED
        assert not missing, (
            f"Methods used in rules but missing from METRIC_ALLOWED_METHODS: "
            f"{sorted(missing)}. These methods will be silently rejected by "
            f"_is_allowed() / _all_allowed() for every risk metric."
        )

    def test_expected_methods_present(self):
        """Sanity check: the extraction finds the methods we know exist."""
        rule_methods = _all_rule_methods()
        for expected in ['kANON', 'LOCSUPR', 'PRAM', 'NOISE',
                         'GENERALIZE', 'GENERALIZE_FIRST']:
            assert expected in rule_methods, (
                f"Expected method {expected!r} not found in rule files — "
                f"extraction may be broken"
            )
