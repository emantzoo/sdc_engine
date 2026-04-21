"""
Canonical rule registry for Spec 22 systemic tests.

Single source of truth for the complete rule inventory.
Update this file when rules are added, renamed, or deleted.
Tests 1 (coverage), 2 (ordering), and 5 (metric appropriateness) import from here.

Cross-reference: rule factories live in sdc_engine/sdc/selection/rules.py
and sdc_engine/sdc/selection/pipelines.py.  The factory evaluation order
is defined in pipelines.py:select_method_suite (lines 474-490).
"""

# ---------------------------------------------------------------------------
# Factory evaluation order in select_method_suite
# ---------------------------------------------------------------------------
# pipeline_rules is evaluated BEFORE the loop (line 413), then these 14 are
# iterated in order (lines 474-490).  Total: 15 factories + 1 pre-check.
FACTORY_ORDER = [
    'pipeline_rules',              # Pre-loop check (GEO1, DYN, P5)
    'regulatory_compliance_rules', # REG1
    'data_structure_rules',        # Always False for microdata
    'small_dataset_rules',         # HR6
    'structural_risk_rules',       # SR3
    'risk_concentration_rules',    # RC1
    'public_release_rules',        # PUB1
    'secure_environment_rules',    # SEC1
    'categorical_aware_rules',     # CAT1
    'l_diversity_rules',           # LDIV1
    'temporal_dominant_rules',     # DATE1
    'reid_risk_rules',             # QR0-QR4, MED1
    'low_risk_rules',              # LOW1-LOW3
    'distribution_rules',          # DP1-DP3 (dead by position — after LOW3 catch-all)
    'uniqueness_risk_rules',       # HR1-HR5 (reachable only when has_reid=False)
    'default_rules',               # DEFAULT_* (reachable only after all above fail)
]

# The factory loop order (excluding pipeline_rules pre-check)
FACTORY_LOOP_ORDER = FACTORY_ORDER[1:]

# ---------------------------------------------------------------------------
# Canonical rule inventory
# ---------------------------------------------------------------------------
# Key:   rule name as it appears in result['rule'] or suite['rule_applied']
# Value: dict with:
#   method   — primary method (str), list (pipeline), or None (dynamic)
#   factory  — factory function name
#   is_pipeline — True for pipeline rules (optional, default False)
#
# Suppression-gated variants (e.g. QR3_Uniform_High_Risk_Clamped) are
# covered via SUPPRESSION_GATED_PREFIXES — tests match on prefix.
# ---------------------------------------------------------------------------

CANONICAL_RULES = {
    # --- Pipeline rules (pipelines.py) ---
    'GEO1_Multi_Level_Geographic': {
        'method': ['GENERALIZE', 'kANON'],
        'factory': 'pipeline_rules',
        'is_pipeline': True,
    },
    'DYN_Pipeline': {
        'method': None,  # dynamic — varies by data shape
        'factory': 'pipeline_rules',
        'is_pipeline': True,
    },
    'P5_Small_Dataset_Mixed_Risks': {
        'method': ['NOISE', 'PRAM'],
        'factory': 'pipeline_rules',
        'is_pipeline': True,
    },

    # --- Regulatory compliance ---
    'REG1_Regulatory_High_Risk': {
        'method': 'kANON',
        'factory': 'regulatory_compliance_rules',
    },
    'REG1_Regulatory_Moderate_Risk': {
        'method': 'kANON',
        'factory': 'regulatory_compliance_rules',
    },

    # --- data_structure_rules has NO sub-rules (always returns False) ---

    # --- Small dataset ---
    'HR6_Very_Small_Dataset': {
        'method': 'LOCSUPR',
        'factory': 'small_dataset_rules',
    },

    # --- Structural risk ---
    'SR3_Near_Unique_Few_QIs': {
        'method': 'LOCSUPR',
        'factory': 'structural_risk_rules',
    },

    # --- Risk concentration ---
    'RC1_Risk_Dominated': {
        'method': 'LOCSUPR',
        'factory': 'risk_concentration_rules',
    },

    # --- Public release ---
    'PUB1_Public_Release_High_Risk': {
        'method': 'kANON',
        'factory': 'public_release_rules',
    },
    'PUB1_Public_Release_Moderate_Risk': {
        'method': 'kANON',
        'factory': 'public_release_rules',
    },

    # --- Secure environment ---
    'SEC1_Secure_Categorical': {
        'method': 'PRAM',
        'factory': 'secure_environment_rules',
    },
    'SEC1_Secure_Continuous': {
        'method': 'NOISE',
        'factory': 'secure_environment_rules',
    },

    # --- Categorical-aware ---
    'CAT1_Categorical_Dominant': {
        'method': 'PRAM',
        'factory': 'categorical_aware_rules',
    },

    # --- L-diversity ---
    'LDIV1_Low_Sensitive_Diversity': {
        'method': 'PRAM',
        'factory': 'l_diversity_rules',
    },
    'LDIV1_DATE1_Merged': {
        'method': 'PRAM',
        'factory': 'l_diversity_rules',
    },

    # --- Temporal-dominant ---
    'DATE1_Temporal_Dominant': {
        'method': 'PRAM',
        'factory': 'temporal_dominant_rules',
    },

    # --- ReID risk ---
    'QR0_K_Anonymity_Infeasible': {
        'method': 'GENERALIZE_FIRST',
        'factory': 'reid_risk_rules',
    },
    'QR1_Severe_Tail_Risk': {
        'method': 'LOCSUPR',
        'factory': 'reid_risk_rules',
    },
    'QR2_Heavy_Tail_Low_Suppression': {
        'method': 'LOCSUPR',
        'factory': 'reid_risk_rules',
    },
    'QR2_Heavy_Tail_Risk': {
        'method': 'kANON',
        'factory': 'reid_risk_rules',
    },
    'QR2_Moderate_Tail_Risk': {
        'method': 'LOCSUPR',
        'factory': 'reid_risk_rules',
    },
    'QR3_Uniform_High_Risk': {
        'method': 'kANON',
        'factory': 'reid_risk_rules',
    },
    'QR4_Widespread_High': {
        'method': 'kANON',
        'factory': 'reid_risk_rules',
    },
    'QR4_Widespread_Moderate': {
        'method': 'kANON',
        'factory': 'reid_risk_rules',
    },
    'MED1_Moderate_Structural': {
        'method': 'kANON',
        'factory': 'reid_risk_rules',
    },

    # --- Low risk ---
    'LOW1_Categorical': {
        'method': 'PRAM',
        'factory': 'low_risk_rules',
    },
    'LOW2_Continuous_Noise': {
        'method': 'NOISE',
        'factory': 'low_risk_rules',
    },
    'LOW2_Continuous_kANON': {
        'method': 'kANON',
        'factory': 'low_risk_rules',
    },
    'LOW3_Mixed': {
        'method': 'kANON',
        'factory': 'low_risk_rules',
    },

    # --- Distribution (dead by position — after LOW3 catch-all) ---
    # These rules are placed after low_risk_rules which has an unconditional
    # catch-all (LOW3).  They are dead code in the chain but can be triggered
    # via direct factory call.  Flagged for Test 2 / Spec 21 cleanup.
    'DP1_Outliers': {
        'method': 'NOISE',
        'factory': 'distribution_rules',
        'dead_by_position': True,
    },
    'DP2_Skewed': {
        'method': 'PRAM',
        'factory': 'distribution_rules',
        'dead_by_position': True,
    },
    'DP3_Sensitive': {
        'method': 'kANON',
        'factory': 'distribution_rules',
        'dead_by_position': True,
    },

    # --- Uniqueness risk (reachable only when has_reid=False) ---
    'HR1_Extreme_Uniqueness': {
        'method': 'LOCSUPR',
        'factory': 'uniqueness_risk_rules',
    },
    'HR2_Very_High_Uniqueness': {
        'method': 'kANON',
        'factory': 'uniqueness_risk_rules',
    },
    'HR3_High_Uniqueness_QIs': {
        'method': 'kANON',
        'factory': 'uniqueness_risk_rules',
    },
    'HR4_Very_Small_Dataset': {
        'method': 'PRAM',
        'factory': 'uniqueness_risk_rules',
    },
    'HR5_Small_Dataset': {
        'method': 'NOISE',  # NOISE when n_continuous > 0, PRAM otherwise
        'factory': 'uniqueness_risk_rules',
    },

    # --- Default rules ---
    'DEFAULT_Microdata_QIs': {
        'method': 'kANON',
        'factory': 'default_rules',
    },
    'DEFAULT_Categorical': {
        'method': 'PRAM',
        'factory': 'default_rules',
    },
    'DEFAULT_Continuous': {
        'method': 'NOISE',
        'factory': 'default_rules',
    },
    'DEFAULT_Fallback': {
        'method': 'PRAM',
        'factory': 'default_rules',
    },

}


# Suppression-gated rule name prefixes.
# _suppression_gated_kanon() can append _Clamped, _Supp_Switch_PRAM,
# or _Supp_Switch_LOCSUPR to any of these base rule names.
SUPPRESSION_GATED_PREFIXES = [
    'QR3_Uniform_High_Risk',
    'QR4_Widespread_High',
    'QR4_Widespread_Moderate',
    'MED1_Moderate_Structural',
]
SUPPRESSION_SUFFIXES = ['_Clamped', '_Supp_Switch_PRAM', '_Supp_Switch_LOCSUPR']
