"""Spec 22 Test 1 — Positive-path rule coverage.

For every rule in the canonical registry, verify there exists a features
dict that makes the rule fire (applies=True) and return the expected
rule name.  This catches:
  - Dead rules (gate conditions that can never be satisfied)
  - Renamed rules (canonical list drift from code)
  - New rules missing from the canonical list (staleness guard)

DP1-DP3 note: These rules are dead-by-position (placed after LOW3
unconditional catch-all in low_risk_rules).  They are tested here via
direct factory calls, bypassing the chain.  Their dead-by-position
status is validated in Test 2 (rule ordering) and flagged for Spec 21
cleanup.  See docs/investigations/unconditional_catchall_audit.md.
"""

import ast
import pytest
from pathlib import Path

from sdc_engine.sdc.selection.rules import (
    regulatory_compliance_rules,
    data_structure_rules,
    small_dataset_rules,
    structural_risk_rules,
    risk_concentration_rules,
    public_release_rules,
    secure_environment_rules,
    categorical_aware_rules,
    l_diversity_rules,
    temporal_dominant_rules,
    reid_risk_rules,
    low_risk_rules,
    distribution_rules,
    uniqueness_risk_rules,
    default_rules,
)
from sdc_engine.sdc.selection import pipelines

from tests.fixtures.canonical_rules import (
    CANONICAL_RULES,
    SUPPRESSION_GATED_PREFIXES,
    SUPPRESSION_SUFFIXES,
)


# ---------------------------------------------------------------------------
# Feature key inventory
# ---------------------------------------------------------------------------
# Cross-referenced against Spec 19 Phase 1.1 feature diff table (41 keys)
# plus keys accessed by select_method_suite itself (_access_tier, _audit,
# method_constraints) and keys consumed by helper functions
# (qi_max_category_freq, high_cardinality_count, geo_qis_by_granularity,
# estimated_suppression_k5, recommended_qi_to_remove, min_l, cat_ratio,
# high_card_count).  Total: ~53 keys.
#
# Keys accessed via features['key'] (KeyError if missing):
#   quasi_identifiers, data_type, n_categorical, n_continuous,
#   continuous_vars, risk_pattern, uniqueness_rate, n_qis, n_records,
#   has_outliers, skewed_columns, has_sensitive_attributes,
#   reid_50, reid_95, reid_99, high_risk_rate
#
# Keys accessed via features.get('key', default):
#   has_reid, max_qi_uniqueness, var_priority, risk_concentration,
#   k_anonymity_feasibility, estimated_suppression, qi_cardinalities,
#   high_risk_rate, high_cardinality_count, qi_max_category_freq,
#   _risk_metric_type, _access_tier, _utility_floor, _reid_target_raw,
#   qi_type_counts, geo_qis_by_granularity, sensitive_column_diversity,
#   min_l, sensitive_columns, date_columns, qi_cardinality_product,
#   expected_eq_size, estimated_suppression_k5, cat_ratio, high_card_count,
#   bimodal_risk, mean_risk, max_risk, recommended_qi_to_remove,
#   method_constraints, qi_treatment


def _base_features(**overrides):
    """Complete features dict with safe defaults for all ~53 keys.

    Defaults produce a moderate-risk microdata profile (reid_95=0.25,
    widespread pattern) that reaches reid_risk_rules if no earlier
    rule's gates are satisfied.
    """
    base = {
        # --- Core dataset properties ---
        'data_type': 'microdata',
        'n_records': 5000,
        'n_columns': 5,
        'quasi_identifiers': ['age', 'sex', 'education'],
        'n_qis': 3,

        # --- Variable types ---
        'n_continuous': 1,
        'n_categorical': 2,
        'continuous_vars': ['age'],
        'categorical_vars': ['sex', 'education'],
        'cat_ratio': 0.67,
        'high_cardinality_qis': [],
        'low_cardinality_qis': ['sex', 'education'],
        'high_cardinality_count': 0,
        'high_card_count': 0,

        # --- ReID metrics ---
        'has_reid': True,
        'reid_50': 0.15,
        'reid_95': 0.25,
        'reid_99': 0.35,
        'mean_risk': 0.18,
        'max_risk': 0.50,
        'high_risk_count': 400,
        'high_risk_rate': 0.08,

        # --- Risk classification ---
        'risk_pattern': 'moderate',
        'bimodal_risk': False,

        # --- Uniqueness ---
        'uniqueness_rate': 0.05,
        'max_qi_uniqueness': 0.30,

        # --- Distribution ---
        'has_outliers': False,
        'skewed_columns': [],

        # --- Feasibility ---
        'k_anonymity_feasibility': 'feasible',
        'qi_cardinality_product': 1000,
        'expected_eq_size': 5.0,
        'estimated_suppression': {},
        'estimated_suppression_k5': 0.10,

        # --- QI metadata ---
        'qi_cardinalities': {'age': 60, 'sex': 2, 'education': 10},
        'qi_max_category_freq': {'sex': 0.52, 'education': 0.15},
        'qi_type_counts': {},
        'geo_qis_by_granularity': {},
        'date_columns': [],

        # --- Sensitivity ---
        'has_sensitive_attributes': False,
        'sensitive_columns': {},
        'sensitive_column_diversity': None,
        'min_l': None,

        # --- Context (injected by select_method_suite) ---
        '_risk_metric_type': 'reid95',
        '_access_tier': 'SCIENTIFIC',
        '_utility_floor': 0.80,
        '_reid_target_raw': None,

        # --- var_priority (stripped by default — RC rules stay dormant) ---
        'var_priority': {},
        'risk_concentration': {},

        # --- User constraints ---
        'method_constraints': {},
        'qi_treatment': None,

        # --- Other ---
        'recommended_qi_to_remove': None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Factory function lookup
# ---------------------------------------------------------------------------

_FACTORY_MAP = {
    'pipeline_rules': pipelines.pipeline_rules,
    'regulatory_compliance_rules': regulatory_compliance_rules,
    'data_structure_rules': data_structure_rules,
    'small_dataset_rules': small_dataset_rules,
    'structural_risk_rules': structural_risk_rules,
    'risk_concentration_rules': risk_concentration_rules,
    'public_release_rules': public_release_rules,
    'secure_environment_rules': secure_environment_rules,
    'categorical_aware_rules': categorical_aware_rules,
    'l_diversity_rules': l_diversity_rules,
    'temporal_dominant_rules': temporal_dominant_rules,
    'reid_risk_rules': reid_risk_rules,
    'low_risk_rules': low_risk_rules,
    'distribution_rules': distribution_rules,
    'uniqueness_risk_rules': uniqueness_risk_rules,
    'default_rules': default_rules,
}


# ---------------------------------------------------------------------------
# Feature overrides per rule
# ---------------------------------------------------------------------------

_RULE_FEATURES = {
    # --- Pipeline rules ---
    'GEO1_Multi_Level_Geographic': {
        'has_reid': True,
        'reid_95': 0.25,
        # build_dynamic_pipeline requires reid_95 > 0.15 and checks GEO1 first.
        # Categorical guard: n_cat/(n_cat+n_cont) < 0.70 needed.
        'geo_qis_by_granularity': {'zipcode': 'fine', 'region': 'coarse'},
        'quasi_identifiers': ['zipcode', 'region', 'age', 'income'],
        'n_qis': 4,
        'n_continuous': 2,  # 2/(2+2) = 0.50 < 0.70
        'n_categorical': 2,
        'cat_ratio': 0.50,
        'k_anonymity_feasibility': 'feasible',
        'estimated_suppression': {3: 0.10},
    },
    'DYN_Pipeline': {
        # DYN needs len(pipeline) >= 2.  With reid_95 > 0.20 kANON is added,
        # then NOISE is skipped (kANON already in pipeline), so LOCSUPR must
        # also be added: needs high_risk_rate > 0.30 AND est_supp_k7 < 0.40.
        'reid_95': 0.45,
        'has_reid': True,
        'n_continuous': 2,
        'n_categorical': 2,
        'continuous_vars': ['age', 'income'],
        'has_outliers': True,
        'high_risk_rate': 0.35,
        'cat_ratio': 0.50,
        'k_anonymity_feasibility': 'feasible',
        'estimated_suppression': {3: 0.10, 5: 0.15, 7: 0.20},
    },
    'P5_Small_Dataset_Mixed_Risks': {
        # P5: density < 5, uniqueness > 0.15, n_cont >= 2, n_cat >= 2, n >= 200
        # DYN must not fire: reid_95 <= 0.15
        'n_records': 500,
        'uniqueness_rate': 0.20,
        'n_continuous': 2,
        'n_categorical': 2,
        'continuous_vars': ['age', 'income'],
        'categorical_vars': ['sex', 'education'],
        'qi_cardinality_product': 120,  # density = 500/120 ≈ 4.17 < 5
        'reid_95': 0.10,  # low reid so DYN won't fire (guard: reid_95 <= 0.15)
        'has_reid': True,
        'k_anonymity_feasibility': 'feasible',
        'estimated_suppression': {3: 0.10},
    },

    # --- Regulatory compliance ---
    'REG1_Regulatory_High_Risk': {
        '_access_tier': 'PUBLIC',
        '_reid_target_raw': 0.03,
        'reid_95': 0.20,
    },
    'REG1_Regulatory_Moderate_Risk': {
        '_access_tier': 'PUBLIC',
        '_reid_target_raw': 0.03,
        'reid_95': 0.10,
    },

    # --- Small dataset ---
    'HR6_Very_Small_Dataset': {
        'n_records': 100,
        'n_qis': 3,
    },

    # --- Structural risk ---
    'SR3_Near_Unique_Few_QIs': {
        'has_reid': True,
        'n_qis': 2,
        'quasi_identifiers': ['age', 'income'],
        'max_qi_uniqueness': 0.85,
        'reid_95': 0.30,
        'qi_cardinalities': {'age': 60, 'income': 500},
    },

    # --- Risk concentration ---
    'RC1_Risk_Dominated': {
        'reid_95': 0.20,
        'var_priority': {
            'age': ('HIGH', 65.0),
            'sex': ('LOW', 15.0),
            'education': ('LOW', 12.0),
        },
        'risk_concentration': {
            'pattern': 'dominated',
            'top_qi': 'age',
            'top_pct': 65.0,
            'top2_pct': 80.0,
            'n_high_risk': 1,
        },
        'k_anonymity_feasibility': 'feasible',
    },

    # --- Public release ---
    'PUB1_Public_Release_High_Risk': {
        '_access_tier': 'PUBLIC',
        '_reid_target_raw': 0.01,
        'reid_95': 0.25,
    },
    'PUB1_Public_Release_Moderate_Risk': {
        '_access_tier': 'PUBLIC',
        '_reid_target_raw': 0.01,
        'reid_95': 0.15,
    },

    # --- Secure environment ---
    'SEC1_Secure_Categorical': {
        '_access_tier': 'SECURE',
        '_utility_floor': 0.92,
        'reid_95': 0.15,
        'n_categorical': 3,
        'n_continuous': 1,
        'cat_ratio': 0.75,
    },
    'SEC1_Secure_Continuous': {
        '_access_tier': 'SECURE',
        '_utility_floor': 0.92,
        'reid_95': 0.15,
        'n_categorical': 1,
        'n_continuous': 2,
        'continuous_vars': ['age', 'income'],
        'cat_ratio': 0.33,
    },

    # --- Categorical-aware ---
    'CAT1_Categorical_Dominant': {
        '_risk_metric_type': 'l_diversity',
        'has_reid': True,
        'reid_95': 0.25,
        'n_categorical': 4,
        'n_continuous': 1,
        'categorical_vars': ['sex', 'education', 'region', 'job'],
        'cat_ratio': 0.80,
        'qi_max_category_freq': {'sex': 0.52, 'education': 0.15, 'region': 0.10, 'job': 0.08},
    },

    # --- L-diversity ---
    'LDIV1_Low_Sensitive_Diversity': {
        '_risk_metric_type': 'reid95',
        'reid_95': 0.08,
        'has_reid': True,
        'sensitive_column_diversity': 3,
        'min_l': 1,
        'has_sensitive_attributes': True,
        'sensitive_columns': {'income': 'financial'},
        'k_anonymity_feasibility': 'feasible',
        'estimated_suppression': {3: 0.10},
        'qi_type_counts': {},  # no dates — avoid LDIV1_DATE1_Merged
    },
    'LDIV1_DATE1_Merged': {
        '_risk_metric_type': 'reid95',
        'reid_95': 0.08,
        'has_reid': True,
        'sensitive_column_diversity': 3,
        'min_l': 1,
        'has_sensitive_attributes': True,
        'sensitive_columns': {'income': 'financial'},
        'k_anonymity_feasibility': 'feasible',
        'estimated_suppression': {3: 0.10},
        'n_qis': 4,
        'quasi_identifiers': ['birth_date', 'hire_date', 'sex', 'education'],
        'qi_type_counts': {'date': 2, 'categorical': 2},
    },

    # --- Temporal-dominant ---
    'DATE1_Temporal_Dominant': {
        'n_qis': 4,
        'quasi_identifiers': ['birth_date', 'hire_date', 'exit_date', 'sex'],
        'qi_type_counts': {'date': 3, 'categorical': 1},
        'reid_95': 0.25,
        'has_reid': True,
    },

    # --- ReID risk ---
    'QR0_K_Anonymity_Infeasible': {
        'has_reid': True,
        'k_anonymity_feasibility': 'infeasible',
        'expected_eq_size': 1.5,
        'qi_cardinality_product': 5000,
        'reid_95': 0.50,
        'reid_50': 0.30,
        'reid_99': 0.80,
        'risk_pattern': 'uniform_high',
    },
    'QR1_Severe_Tail_Risk': {
        'has_reid': True,
        'risk_pattern': 'severe_tail',
        'reid_50': 0.02,
        'reid_95': 0.35,
        'reid_99': 0.90,
        'k_anonymity_feasibility': 'feasible',
    },
    'QR2_Heavy_Tail_Low_Suppression': {
        'has_reid': True,
        'risk_pattern': 'tail',
        'reid_95': 0.45,
        'reid_50': 0.10,
        'reid_99': 0.60,
        'estimated_suppression': {7: 0.30, 5: 0.20, 3: 0.10},
        'k_anonymity_feasibility': 'feasible',
    },
    'QR2_Heavy_Tail_Risk': {
        'has_reid': True,
        'risk_pattern': 'tail',
        'reid_95': 0.45,
        'reid_50': 0.10,
        'reid_99': 0.60,
        'estimated_suppression': {7: 0.10, 5: 0.08, 3: 0.05},
        'k_anonymity_feasibility': 'feasible',
    },
    'QR2_Moderate_Tail_Risk': {
        'has_reid': True,
        'risk_pattern': 'tail',
        'reid_95': 0.35,
        'reid_50': 0.10,
        'reid_99': 0.50,
        'k_anonymity_feasibility': 'feasible',
    },
    'QR3_Uniform_High_Risk': {
        'has_reid': True,
        'risk_pattern': 'uniform_high',
        'reid_50': 0.25,
        'reid_95': 0.30,
        'reid_99': 0.32,
        'k_anonymity_feasibility': 'feasible',
        'estimated_suppression': {10: 0.15, 7: 0.10, 5: 0.05, 3: 0.02},
    },
    'QR4_Widespread_High': {
        'has_reid': True,
        'risk_pattern': 'widespread',
        'reid_50': 0.25,
        'reid_95': 0.55,
        'reid_99': 0.70,
        'k_anonymity_feasibility': 'feasible',
        'estimated_suppression': {10: 0.15, 7: 0.10, 5: 0.05, 3: 0.02},
    },
    'QR4_Widespread_Moderate': {
        'has_reid': True,
        'risk_pattern': 'widespread',
        'reid_50': 0.20,
        'reid_95': 0.40,
        'reid_99': 0.50,
        'k_anonymity_feasibility': 'feasible',
        'estimated_suppression': {7: 0.10, 5: 0.05, 3: 0.02},
    },
    'MED1_Moderate_Structural': {
        'has_reid': True,
        'reid_95': 0.25,
        'reid_50': 0.05,
        'reid_99': 0.40,
        'risk_pattern': 'moderate',
        'high_risk_rate': 0.12,
        'k_anonymity_feasibility': 'feasible',
        'estimated_suppression': {5: 0.10, 3: 0.05},
    },

    # --- Low risk ---
    'LOW1_Categorical': {
        'has_reid': True,
        'reid_95': 0.08,
        'n_categorical': 3,
        'n_continuous': 0,
        'continuous_vars': [],
        'categorical_vars': ['sex', 'education', 'region'],
        'cat_ratio': 1.0,
        'high_cardinality_count': 0,
        'high_card_count': 0,
        'risk_pattern': 'uniform_low',
    },
    'LOW2_Continuous_Noise': {
        'has_reid': True,
        'reid_95': 0.04,
        'n_categorical': 1,
        'n_continuous': 2,
        'continuous_vars': ['age', 'income'],
        'categorical_vars': ['sex'],
        'cat_ratio': 0.33,
        'has_outliers': False,
        'risk_pattern': 'uniform_low',
    },
    'LOW2_Continuous_kANON': {
        'has_reid': True,
        'reid_95': 0.12,
        'n_categorical': 1,
        'n_continuous': 2,
        'continuous_vars': ['age', 'income'],
        'categorical_vars': ['sex'],
        'cat_ratio': 0.33,
        'has_outliers': False,
        'risk_pattern': 'uniform_low',
    },
    'LOW3_Mixed': {
        'has_reid': True,
        'reid_95': 0.08,
        'n_categorical': 2,
        'n_continuous': 2,
        'continuous_vars': ['age', 'income'],
        'categorical_vars': ['sex', 'education'],
        'cat_ratio': 0.50,
        'high_cardinality_count': 0,
        'high_card_count': 0,
        'risk_pattern': 'uniform_low',
    },

    # --- Distribution (dead by position — tested via direct factory call) ---
    'DP1_Outliers': {
        'has_outliers': True,
        'n_continuous': 2,
        'continuous_vars': ['age', 'income'],
    },
    'DP2_Skewed': {
        'skewed_columns': ['age', 'income'],
        'has_outliers': False,
        'n_continuous': 0,
        'continuous_vars': [],
    },
    'DP3_Sensitive': {
        'has_sensitive_attributes': True,
        'n_qis': 3,
        'skewed_columns': [],
        'has_outliers': False,
        'n_continuous': 0,
        'continuous_vars': [],
    },

    # --- Uniqueness risk (has_reid=False path) ---
    'HR1_Extreme_Uniqueness': {
        'uniqueness_rate': 0.25,
        'n_qis': 3,
        'n_records': 5000,
    },
    'HR2_Very_High_Uniqueness': {
        'uniqueness_rate': 0.15,
        'n_qis': 3,
        'n_records': 5000,
    },
    'HR3_High_Uniqueness_QIs': {
        'uniqueness_rate': 0.08,
        'n_qis': 3,
        'n_records': 5000,
    },
    'HR4_Very_Small_Dataset': {
        'uniqueness_rate': 0.02,
        'n_qis': 3,
        'n_records': 50,
    },
    'HR5_Small_Dataset': {
        'uniqueness_rate': 0.05,
        'n_qis': 3,
        'n_records': 200,
        'n_continuous': 1,
        'continuous_vars': ['age'],
    },

    # --- Default rules ---
    'DEFAULT_Microdata_QIs': {
        'data_type': 'microdata',
        'n_qis': 3,
        # Must bypass all earlier rules: has_reid=False, low uniqueness,
        # no sensitive attrs, no outliers, no skew, n_records >= 500
        'has_reid': False,
        'uniqueness_rate': 0.01,
        'n_records': 5000,
        'has_outliers': False,
        'skewed_columns': [],
        'has_sensitive_attributes': False,
    },
    'DEFAULT_Categorical': {
        'data_type': 'other',
        'n_qis': 1,
        'n_categorical': 3,
        'n_continuous': 1,
    },
    'DEFAULT_Continuous': {
        'data_type': 'other',
        'n_qis': 1,
        'n_categorical': 1,
        'n_continuous': 3,
        'continuous_vars': ['a', 'b', 'c'],
    },
    'DEFAULT_Fallback': {
        'data_type': 'other',
        'n_qis': 1,
        'n_categorical': 2,
        'n_continuous': 2,
        'continuous_vars': ['a', 'b'],
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Rules that are NOT directly callable via their factory function because
# they need the full chain or special handling
_SKIP_DIRECT_FACTORY = {
    # DYN_Pipeline needs pipeline_rules -> build_dynamic_pipeline path
    # Tested separately below
}


class TestPositivePathCoverage:
    """Every rule in CANONICAL_RULES must fire when given appropriate features."""

    @pytest.mark.parametrize("rule_name", sorted(
        r for r in CANONICAL_RULES
        if r != 'DYN_Pipeline'  # tested separately
    ))
    def test_rule_fires(self, rule_name):
        """Rule fires (applies=True) with crafted feature overrides."""
        info = CANONICAL_RULES[rule_name]
        factory_name = info['factory']
        factory_fn = _FACTORY_MAP[factory_name]

        overrides = _RULE_FEATURES.get(rule_name, {})
        features = _base_features(**overrides)

        result = factory_fn(features)

        assert result.get('applies'), (
            f"Rule {rule_name} did not fire.\n"
            f"  Factory: {factory_name}\n"
            f"  Overrides: {overrides}\n"
            f"  Result: {result}"
        )

        actual_rule = result.get('rule', '')
        # Suppression-gated variants may append suffixes to the base name
        if rule_name in SUPPRESSION_GATED_PREFIXES:
            assert actual_rule.startswith(rule_name), (
                f"Expected rule starting with '{rule_name}', got '{actual_rule}'"
            )
        else:
            assert actual_rule == rule_name, (
                f"Expected rule '{rule_name}', got '{actual_rule}'"
            )

    def test_dyn_pipeline_fires(self):
        """DYN_Pipeline fires via pipeline_rules → build_dynamic_pipeline."""
        overrides = _RULE_FEATURES['DYN_Pipeline']
        features = _base_features(**overrides)

        result = pipelines.pipeline_rules(features)

        assert result.get('applies'), (
            f"DYN_Pipeline did not fire.\n"
            f"  Result: {result}"
        )
        assert result.get('rule') == 'DYN_Pipeline'


class TestP5Density:
    """P5 requires density < 5.  Verify our fixture achieves that."""

    def test_p5_density_below_threshold(self):
        overrides = _RULE_FEATURES['P5_Small_Dataset_Mixed_Risks']
        features = _base_features(**overrides)
        n = features['n_records']
        qcp = features['qi_cardinality_product']
        density = n / qcp if qcp > 0 else float('inf')
        assert density < 5, f"P5 requires density < 5, got {density}"


class TestStalenessGuard:
    """Rule names in source code must be a subset of CANONICAL_RULES."""

    _SELECTION_DIR = (
        Path(__file__).resolve().parents[1]
        / 'sdc_engine' / 'sdc' / 'selection'
    )

    @staticmethod
    def _extract_rule_literals(filepath):
        """Extract all string values of 'rule' dict keys from AST."""
        source = filepath.read_text(encoding='utf-8')
        tree = ast.parse(source)
        rules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (isinstance(key, ast.Constant)
                            and key.value == 'rule'
                            and isinstance(value, ast.Constant)
                            and isinstance(value.value, str)):
                        rules.add(value.value)
        return rules

    # EMERGENCY_FALLBACK is an inline catch-all in select_method_by_features
    # and select_method_suite, not a rule factory output.
    _KNOWN_NON_FACTORY_RULES = {'EMERGENCY_FALLBACK'}

    def test_rules_py_names_in_canonical(self):
        """All rule name literals in rules.py are in CANONICAL_RULES."""
        found = self._extract_rule_literals(self._SELECTION_DIR / 'rules.py')
        canonical = set(CANONICAL_RULES.keys()) | self._KNOWN_NON_FACTORY_RULES
        unknown = found - canonical
        assert not unknown, (
            f"rules.py contains rule names not in CANONICAL_RULES: {unknown}\n"
            f"Update tests/fixtures/canonical_rules.py to include them."
        )

    def test_pipelines_py_names_in_canonical(self):
        """All rule name literals in pipelines.py are in CANONICAL_RULES."""
        found = self._extract_rule_literals(
            self._SELECTION_DIR / 'pipelines.py')
        canonical = set(CANONICAL_RULES.keys())
        unknown = found - canonical
        assert not unknown, (
            f"pipelines.py contains rule names not in CANONICAL_RULES: {unknown}\n"
            f"Update tests/fixtures/canonical_rules.py to include them."
        )
