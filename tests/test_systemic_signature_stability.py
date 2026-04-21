"""
Spec 22 — Test 6: Signature Stability for Gatekeeper Functions
==============================================================

Validates that key functions acting as gatekeepers between components
have stable signatures.  New parameters must be keyword-only with defaults.

Gatekeeper functions:
    - build_data_features (protection_engine.py)
    - select_method_suite (pipelines.py)
    - calculate_reid (metrics/reid.py)

This test snapshots the current signatures and fails if:
    - Positional parameters are removed or reordered
    - Required parameters are added (no default)
    - Existing keyword-only parameters lose their defaults

Non-breaking changes pass:
    - New keyword-only parameters with defaults
    - Default value changes (documented as intentional)
"""
import inspect
import pytest

from sdc_engine.sdc.protection_engine import build_data_features
from sdc_engine.sdc.selection.pipelines import select_method_suite
from sdc_engine.sdc.metrics.reid import calculate_reid


# ---------------------------------------------------------------------------
# Signature snapshots (current as of Spec 22)
# ---------------------------------------------------------------------------
# Format: {'positional': [(name, has_default), ...],
#           'keyword_only': [(name, has_default), ...]}

GATEKEEPER_SNAPSHOTS = {
    'build_data_features': {
        'positional': [
            ('data', False),
            ('quasi_identifiers', False),
            ('reid', True),
            ('active_cols', True),
            ('var_priority', True),
            ('column_types', True),
            ('risk_metric', True),
            ('risk_target_raw', True),
            ('sensitive_columns', True),
            ('preprocess_metadata', True),
        ],
        'keyword_only': [],
    },
    'select_method_suite': {
        'positional': [
            ('features', False),
            ('access_tier', True),
            ('verbose', True),
        ],
        'keyword_only': [],
    },
    'calculate_reid': {
        'positional': [
            ('data', False),
            ('quasi_identifiers', False),
            ('quantiles', True),
        ],
        'keyword_only': [],
    },
}

GATEKEEPER_FUNCTIONS = {
    'build_data_features': build_data_features,
    'select_method_suite': select_method_suite,
    'calculate_reid': calculate_reid,
}


def _extract_signature(func):
    """Extract positional and keyword-only params from a function signature."""
    sig = inspect.signature(func)
    positional = []
    keyword_only = []
    for name, param in sig.parameters.items():
        has_default = param.default is not inspect.Parameter.empty
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional.append((name, has_default))
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            keyword_only.append((name, has_default))
    return {'positional': positional, 'keyword_only': keyword_only}


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSignatureStability:

    @pytest.mark.parametrize("func_name", list(GATEKEEPER_FUNCTIONS.keys()))
    def test_positional_params_stable(self, func_name):
        """Positional parameters must not be removed or reordered."""
        func = GATEKEEPER_FUNCTIONS[func_name]
        snapshot = GATEKEEPER_SNAPSHOTS[func_name]
        current = _extract_signature(func)

        # Every positional param in the snapshot must appear in the same
        # position in the current signature
        for i, (snap_name, snap_has_default) in enumerate(snapshot['positional']):
            assert i < len(current['positional']), (
                f"{func_name}: positional param '{snap_name}' (pos {i}) "
                f"was removed. Current has only {len(current['positional'])} params."
            )
            cur_name, cur_has_default = current['positional'][i]
            assert cur_name == snap_name, (
                f"{func_name}: positional param at pos {i} changed from "
                f"'{snap_name}' to '{cur_name}'"
            )

    @pytest.mark.parametrize("func_name", list(GATEKEEPER_FUNCTIONS.keys()))
    def test_no_new_required_params(self, func_name):
        """No new required (no-default) parameters added."""
        func = GATEKEEPER_FUNCTIONS[func_name]
        snapshot = GATEKEEPER_SNAPSHOTS[func_name]
        current = _extract_signature(func)

        # Existing params are checked above. New params (beyond snapshot length)
        # must have defaults.
        snap_positional_names = {n for n, _ in snapshot['positional']}
        snap_kwonly_names = {n for n, _ in snapshot['keyword_only']}

        for name, has_default in current['positional']:
            if name not in snap_positional_names:
                assert has_default, (
                    f"{func_name}: new positional param '{name}' has no default. "
                    f"New params must be keyword-only with defaults."
                )

        for name, has_default in current['keyword_only']:
            if name not in snap_kwonly_names:
                assert has_default, (
                    f"{func_name}: new keyword-only param '{name}' has no default"
                )

    @pytest.mark.parametrize("func_name", list(GATEKEEPER_FUNCTIONS.keys()))
    def test_existing_defaults_not_removed(self, func_name):
        """Parameters that had defaults must still have defaults."""
        func = GATEKEEPER_FUNCTIONS[func_name]
        snapshot = GATEKEEPER_SNAPSHOTS[func_name]
        current = _extract_signature(func)

        current_lookup = {n: d for n, d in current['positional']}
        current_lookup.update({n: d for n, d in current['keyword_only']})

        for name, had_default in snapshot['positional'] + snapshot['keyword_only']:
            if had_default and name in current_lookup:
                assert current_lookup[name], (
                    f"{func_name}: param '{name}' lost its default value"
                )
