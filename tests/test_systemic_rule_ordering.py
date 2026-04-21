"""
Spec 22 — Test 2: Rule Chain Ordering
======================================

Validates that:
    1. Factory evaluation order in code matches FACTORY_ORDER in canonical registry
    2. Priority dominance: higher-priority rules win when both match
    3. EMERGENCY_FALLBACK is positioned last (after all factory rules)
    4. Dead-by-position rules (DP1-DP3) are genuinely preempted by LOW3 catch-all

This is the most sophisticated test — it exercises the chain's ordering
semantics, not just individual rule behavior.
"""
import ast
import pytest
from tests.fixtures.canonical_rules import (
    FACTORY_ORDER, FACTORY_LOOP_ORDER, CANONICAL_RULES,
)


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — AST validation: code factory order matches canonical registry
# ═══════════════════════════════════════════════════════════════════════════

class TestFactoryOrderMatchesCode:
    """Parse pipelines.py, extract rule_factories list, compare to FACTORY_ORDER."""

    @staticmethod
    def _extract_factory_list_from_ast():
        """AST-extract the rule_factories = [...] list from pipelines.py."""
        import sdc_engine.sdc.selection.pipelines as mod
        source_path = mod.__file__
        with open(source_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=source_path)

        # Walk the AST looking for `rule_factories = [...]`
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'rule_factories':
                        if isinstance(node.value, ast.List):
                            names = []
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Name):
                                    names.append(elt.id)
                            return names
        return None

    def test_factory_list_extracted(self):
        """Verify we can AST-extract the factory list."""
        names = self._extract_factory_list_from_ast()
        assert names is not None, (
            "Could not find 'rule_factories = [...]' in pipelines.py"
        )
        assert len(names) > 0

    def test_factory_order_matches(self):
        """Code's rule_factories matches FACTORY_LOOP_ORDER (excluding pipeline_rules)."""
        code_names = self._extract_factory_list_from_ast()
        assert code_names is not None

        # FACTORY_LOOP_ORDER is FACTORY_ORDER[1:] (excluding pipeline_rules pre-check)
        assert len(code_names) == len(FACTORY_LOOP_ORDER), (
            f"Code has {len(code_names)} factories, "
            f"canonical has {len(FACTORY_LOOP_ORDER)}. "
            f"Code: {code_names}, Canonical: {FACTORY_LOOP_ORDER}"
        )
        for i, (code_name, canon_name) in enumerate(zip(code_names, FACTORY_LOOP_ORDER)):
            assert code_name == canon_name, (
                f"Position {i}: code has '{code_name}', "
                f"canonical has '{canon_name}'"
            )

    def test_every_canonical_factory_in_code(self):
        """Every factory in FACTORY_ORDER has a corresponding function."""
        from sdc_engine.sdc.selection import rules, pipelines
        for factory_name in FACTORY_ORDER:
            if factory_name == 'pipeline_rules':
                # pipeline_rules is build_dynamic_pipeline + _legacy_pipeline_rules
                assert hasattr(pipelines, 'build_dynamic_pipeline')
                assert hasattr(pipelines, '_legacy_pipeline_rules')
            else:
                assert hasattr(rules, factory_name), (
                    f"Factory '{factory_name}' not found in rules.py"
                )


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Priority dominance: higher-priority wins when both match
# ═══════════════════════════════════════════════════════════════════════════

class TestPriorityDominance:
    """For overlapping rule pairs, verify higher-priority wins.

    Uses full-chain select_method_suite to confirm first-match-wins behavior.
    """

    @staticmethod
    def _select(features_overrides, verbose=False, access_tier='SCIENTIFIC'):
        """Run select_method_suite with features overrides.

        Note: select_method_suite overwrites features['_access_tier'] from
        its access_tier parameter (line 347). Pass access_tier explicitly
        for context-aware rules (REG1, PUB1, SEC1).
        """
        from sdc_engine.sdc.selection.pipelines import select_method_suite
        from tests.test_systemic_rule_coverage import _base_features
        features = _base_features(**features_overrides)
        return select_method_suite(features, access_tier=access_tier, verbose=verbose)

    def test_reg1_beats_pub1(self):
        """REG1 fires before PUB1 when both would match (PUBLIC tier, target=0.03)."""
        suite = self._select({
            '_reid_target_raw': 0.03,
            'reid_95': 0.25, 'reid_50': 0.10, 'reid_99': 0.40,
            'risk_pattern': 'tail',
            'high_risk_rate': 0.15,
        }, access_tier='PUBLIC')
        assert suite['rule_applied'].startswith('REG1'), (
            f"Expected REG1 to beat PUB1, got {suite['rule_applied']}"
        )

    def test_hr6_beats_qr_rules(self):
        """HR6 (small_dataset) fires before QR rules for very small datasets."""
        suite = self._select({
            'n_records': 100,
            'reid_95': 0.50, 'reid_50': 0.20, 'reid_99': 1.0,
            'risk_pattern': 'tail',
            'high_risk_rate': 0.30,
        })
        assert suite['rule_applied'] == 'HR6_Very_Small_Dataset', (
            f"Expected HR6 to beat QR rules, got {suite['rule_applied']}"
        )

    def test_sr3_beats_reid_rules(self):
        """SR3 (structural_risk) fires before reid_risk_rules."""
        suite = self._select({
            'n_qis': 2,
            'max_qi_uniqueness': 0.95,
            'reid_95': 0.50, 'reid_50': 0.20, 'reid_99': 0.80,
            'risk_pattern': 'tail',
            'high_risk_rate': 0.25,
        })
        assert suite['rule_applied'] == 'SR3_Near_Unique_Few_QIs', (
            f"Expected SR3 to beat QR rules, got {suite['rule_applied']}"
        )

    def test_rc1_beats_qr2(self):
        """RC1 (risk_concentration) fires before QR2 when var_priority dominated."""
        suite = self._select({
            'var_priority': {
                'job': ('HIGH', 55.0),
                'sex': ('LOW', 2.0),
                'region': ('LOW', 1.0),
            },
            'risk_concentration': {
                'pattern': 'dominated', 'top_qi': 'job',
                'top_pct': 55.0, 'top2_pct': 57.0, 'n_high_risk': 1,
            },
            'reid_95': 0.35, 'reid_50': 0.15, 'reid_99': 0.60,
            'risk_pattern': 'tail',
            'high_risk_rate': 0.20,
            'n_records': 500,
        })
        assert suite['rule_applied'] == 'RC1_Risk_Dominated', (
            f"Expected RC1 to beat QR2, got {suite['rule_applied']}"
        )

    def test_pub1_beats_cat1(self):
        """PUB1 (public_release) fires before CAT1 when both match."""
        suite = self._select({
            '_reid_target_raw': 0.01,
            'n_categorical': 3, 'n_continuous': 0,
            'categorical_vars': ['a', 'b', 'c'],
            'continuous_vars': [],
            'reid_95': 0.25, 'reid_50': 0.10, 'reid_99': 0.40,
            'risk_pattern': 'tail',
            'high_risk_rate': 0.15,
            '_risk_metric_type': 'l_diversity',
        }, access_tier='PUBLIC')
        assert suite['rule_applied'].startswith('PUB1'), (
            f"Expected PUB1 to beat CAT1, got {suite['rule_applied']}"
        )

    def test_sec1_beats_cat1(self):
        """SEC1 (secure_environment) fires before CAT1 when both match."""
        suite = self._select({
            '_utility_floor': 0.95,
            'n_categorical': 3, 'n_continuous': 0,
            'categorical_vars': ['a', 'b', 'c'],
            'continuous_vars': [],
            'reid_95': 0.15, 'reid_50': 0.05, 'reid_99': 0.25,
            'risk_pattern': 'moderate',
            'high_risk_rate': 0.10,
            '_risk_metric_type': 'l_diversity',
        }, access_tier='SECURE')
        assert suite['rule_applied'].startswith('SEC1'), (
            f"Expected SEC1 to beat CAT1, got {suite['rule_applied']}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 — EMERGENCY_FALLBACK position and reachability
# ═══════════════════════════════════════════════════════════════════════════

class TestEmergencyFallback:
    """Validate EMERGENCY_FALLBACK is positioned last and reachable."""

    def test_fallback_exists_in_code(self):
        """EMERGENCY_FALLBACK string appears in pipelines.py after rule loop."""
        import sdc_engine.sdc.selection.pipelines as mod
        with open(mod.__file__, 'r', encoding='utf-8') as f:
            source = f.read()
        assert 'EMERGENCY_FALLBACK' in source

    def test_fallback_after_rule_loop(self):
        """EMERGENCY_FALLBACK appears after the rule_factories loop in source."""
        import sdc_engine.sdc.selection.pipelines as mod
        with open(mod.__file__, 'r', encoding='utf-8') as f:
            source = f.read()
        loop_pos = source.find('for rule_fn in rule_factories:')
        fallback_pos = source.find("'EMERGENCY_FALLBACK'")
        assert loop_pos > 0, "rule_factories loop not found"
        assert fallback_pos > 0, "EMERGENCY_FALLBACK not found"
        assert fallback_pos > loop_pos, (
            f"EMERGENCY_FALLBACK (pos {fallback_pos}) appears BEFORE "
            f"rule_factories loop (pos {loop_pos})"
        )

    def test_fallback_reachable(self):
        """Construct features where no factory matches → EMERGENCY_FALLBACK fires.

        DEFAULT_Fallback is an unconditional catch-all recommending PRAM.
        Under k_anonymity metric, PRAM is blocked by _is_allowed().
        With data_type='frequency_table' and no QIs, no prior rule fires
        either (all check data_type='microdata' or recommend blocked methods).
        So EMERGENCY_FALLBACK is reached.

        This is the ONLY known path to EMERGENCY_FALLBACK: non-microdata
        data under a metric that blocks PRAM (the DEFAULT_Fallback method).
        """
        from sdc_engine.sdc.selection.pipelines import select_method_suite
        from tests.test_systemic_rule_coverage import _base_features

        features = _base_features(
            data_type='frequency_table',
            n_categorical=0, n_continuous=0,
            categorical_vars=[], continuous_vars=[],
            n_qis=0, quasi_identifiers=[],
            _risk_metric_type='k_anonymity',
        )
        suite = select_method_suite(features, verbose=False)
        assert suite['rule_applied'] == 'EMERGENCY_FALLBACK', (
            f"Expected EMERGENCY_FALLBACK, got {suite['rule_applied']}"
        )

    def test_fallback_output_shape(self):
        """EMERGENCY_FALLBACK output has standard suite keys."""
        from sdc_engine.sdc.selection.pipelines import select_method_suite
        from tests.test_systemic_rule_coverage import _base_features

        features = _base_features(
            data_type='frequency_table',
            n_categorical=0, n_continuous=0,
            categorical_vars=[], continuous_vars=[],
            n_qis=0, quasi_identifiers=[],
            _risk_metric_type='k_anonymity',
        )
        suite = select_method_suite(features, verbose=False)
        assert suite['primary'] == 'kANON'
        assert suite['confidence'] == 'LOW'
        assert isinstance(suite['fallbacks'], list)


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 — Dead-by-position validation
# ═══════════════════════════════════════════════════════════════════════════

class TestDeadByPosition:
    """Validate that rules flagged dead_by_position=True in the canonical
    registry are genuinely preempted by the catch-all above them.

    DP1-DP3 (distribution_rules) are positioned after low_risk_rules in
    the factory chain. low_risk_rules has an unconditional catch-all (LOW3)
    that fires for ANY microdata with has_reid=True and reid_95 <= 0.20.
    distribution_rules also requires has_reid and microdata. So any input
    that passes distribution_rules' entry conditions also passes
    low_risk_rules' entry conditions, and LOW3 catches it first.
    """

    def _get_dead_rules(self):
        return {
            name: info for name, info in CANONICAL_RULES.items()
            if info.get('dead_by_position')
        }

    def test_dead_rules_exist(self):
        """At least DP1-DP3 are flagged."""
        dead = self._get_dead_rules()
        assert len(dead) >= 3
        assert 'DP1_Outliers' in dead
        assert 'DP2_Skewed' in dead
        assert 'DP3_Sensitive' in dead

    def test_dead_rules_in_distribution_factory(self):
        """All dead rules belong to distribution_rules factory."""
        dead = self._get_dead_rules()
        for name, info in dead.items():
            assert info['factory'] == 'distribution_rules', (
                f"Dead rule {name} has factory '{info['factory']}', "
                f"expected 'distribution_rules'"
            )

    def test_distribution_after_low_risk_in_order(self):
        """distribution_rules appears after low_risk_rules in FACTORY_ORDER."""
        low_idx = FACTORY_ORDER.index('low_risk_rules')
        dist_idx = FACTORY_ORDER.index('distribution_rules')
        assert dist_idx > low_idx, (
            f"distribution_rules (pos {dist_idx}) should be after "
            f"low_risk_rules (pos {low_idx})"
        )

    def test_low_risk_preempts_distribution_low_reid(self):
        """When reid_95 <= 0.20 (low-risk territory), low_risk_rules always fires.

        This is the core dead-by-position proof: distribution_rules
        requires reid_95 conditions that overlap with low_risk_rules' range,
        and low_risk_rules fires first with its unconditional catch-all.
        """
        from sdc_engine.sdc.selection.rules import (
            low_risk_rules, distribution_rules,
        )
        from tests.test_systemic_rule_coverage import _base_features

        # DP1_Outliers needs: has_outliers, n_continuous > 0, reid_95 moderate
        features = _base_features(
            has_outliers=True,
            n_continuous=2, n_categorical=1,
            continuous_vars=['x', 'y'],
            categorical_vars=['a'],
            reid_95=0.15, reid_50=0.05, reid_99=0.25,
            risk_pattern='moderate',
        )

        low_result = low_risk_rules(features)
        dp_result = distribution_rules(features)

        # low_risk_rules fires (LOW3 or earlier)
        assert low_result['applies'] is True, (
            "low_risk_rules should fire for this input"
        )
        # distribution_rules also fires (DP1_Outliers)
        assert dp_result['applies'] is True, (
            "distribution_rules should fire for this input"
        )
        # But in the chain, low_risk_rules is evaluated first → DP1 is dead
        # The full chain confirms it:
        from sdc_engine.sdc.selection.pipelines import select_method_suite
        suite = select_method_suite(features, verbose=False)
        assert not suite['rule_applied'].startswith('DP'), (
            f"DP rule '{suite['rule_applied']}' fired through the chain — "
            f"dead-by-position claim is FALSE. This needs Spec 21 investigation."
        )

    def test_low_risk_preempts_all_dp_profiles(self):
        """Sweep multiple profiles that would trigger DP rules.
        Verify low_risk_rules always fires first in the chain."""
        from sdc_engine.sdc.selection.rules import (
            low_risk_rules, distribution_rules,
        )
        from sdc_engine.sdc.selection.pipelines import select_method_suite
        from tests.test_systemic_rule_coverage import _base_features

        profiles = [
            # DP1 territory: outliers + continuous
            {
                'has_outliers': True,
                'n_continuous': 2, 'n_categorical': 1,
                'continuous_vars': ['x', 'y'], 'categorical_vars': ['a'],
                'reid_95': 0.10, 'reid_50': 0.03, 'reid_99': 0.20,
                'skewed_columns': [],
            },
            # DP2 territory: skewed + categorical-dominant
            {
                'skewed_columns': ['x'],
                'n_continuous': 1, 'n_categorical': 2,
                'continuous_vars': ['x'], 'categorical_vars': ['a', 'b'],
                'reid_95': 0.12, 'reid_50': 0.05, 'reid_99': 0.22,
                'has_outliers': False,
            },
            # DP3 territory: has_sensitive
            {
                'has_sensitive_attributes': True,
                'sensitive_columns': {'salary': 'sensitive'},
                'n_continuous': 1, 'n_categorical': 2,
                'continuous_vars': ['x'], 'categorical_vars': ['a', 'b'],
                'reid_95': 0.08, 'reid_50': 0.03, 'reid_99': 0.15,
                'has_outliers': False, 'skewed_columns': [],
            },
        ]

        for i, overrides in enumerate(profiles):
            features = _base_features(**overrides)
            # Confirm distribution_rules would fire
            dp_result = distribution_rules(features)
            if not dp_result.get('applies'):
                continue  # This profile doesn't trigger DP — skip

            # Confirm low_risk_rules fires
            lr_result = low_risk_rules(features)
            assert lr_result['applies'] is True, (
                f"Profile {i}: low_risk_rules should fire"
            )

            # Confirm chain selects LOW, not DP
            suite = select_method_suite(features, verbose=False)
            assert not suite['rule_applied'].startswith('DP'), (
                f"Profile {i}: DP rule '{suite['rule_applied']}' fired through chain"
            )

    def test_dp_rules_callable_directly(self):
        """DP rules can be triggered via direct factory call (bypassing chain).

        This confirms they're functional code (not broken), just unreachable
        through the chain due to position.
        """
        from sdc_engine.sdc.selection.rules import distribution_rules
        from tests.test_systemic_rule_coverage import _base_features, _RULE_FEATURES

        for rule_name in ('DP1_Outliers', 'DP2_Skewed', 'DP3_Sensitive'):
            overrides = _RULE_FEATURES.get(rule_name)
            if overrides is None:
                continue
            features = _base_features(**overrides)
            result = distribution_rules(features)
            assert result['applies'] is True, (
                f"{rule_name} should fire on direct call"
            )
            assert result['rule'] == rule_name
