"""
SDC Preprocessing Interactor
=============================

Bridges the dataset model with SDC's preprocessing pipeline.
Handles data transformations before protection (remove direct IDs,
top/bottom coding, merge rare categories, binning, GENERALIZE).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import logging

from sdc_engine.entities.dataset.base import BaseDataset
from sdc_engine.sdc.utility import (
    compute_utility, compute_per_variable_utility,
    compute_il1s, compute_benchmark_analysis, compute_distributional_metrics,
    compute_composite_utility, compute_fast_qi_utility,
)


@dataclass
class PreprocessingResult:
    """Result of applying SDC preprocessing."""
    preprocessed_data: Optional[pd.DataFrame] = None
    original_data: Optional[pd.DataFrame] = None
    steps_applied: List[str] = field(default_factory=list)
    steps_skipped: List[Dict] = field(default_factory=list)
    step_details: List[Dict] = field(default_factory=list)
    step_utility_progression: List[Dict] = field(default_factory=list)
    reid_before: Optional[float] = None
    reid_after: Optional[float] = None
    utility_score: Optional[float] = None
    per_variable_utility: Optional[Dict] = None
    il1s: Optional[Dict] = None
    benchmark: Optional[Dict] = None
    distributional: Optional[Dict] = None
    comprehensive_metrics: Optional[Dict] = None
    generalize_early_exit: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    recommendations: Dict[str, Any] = field(default_factory=dict)
    smart_defaults: Optional[Dict] = None
    success: bool = False
    error: str = ''


@dataclass
class SDCPreprocessing:
    """Interactor for SDC preprocessing operations on a dataset."""

    dataset: BaseDataset

    def get_recommendations(
        self,
        detected_qis: List[str],
        direct_ids: Dict[str, str],
        reid_95: float,
        preprocessing_report: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate preprocessing recommendations based on data characteristics.

        Returns dict with keys: standard, advanced, all, cache_key.
        """
        df = self.dataset.get()
        preprocessing_report = preprocessing_report or {}

        try:
            from sdc_engine.sdc.recommendations import generate_recommendations
            return generate_recommendations(
                data=df,
                direct_ids=direct_ids,
                detected_qis=detected_qis,
                reid_95=reid_95,
                preprocessing_report=preprocessing_report,
            )
        except Exception as e:
            logging.error(f"Failed to generate recommendations: {e}")
            return {'standard': [], 'advanced': [], 'all': []}

    def get_smart_defaults(
        self,
        detected_qis: List[str],
        initial_reid_95: float,
        structural_risk: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate smart default parameters based on data characteristics.

        Returns dict with keys: preprocess_params, method, method_params, reasoning.
        """
        df = self.dataset.get()
        try:
            from sdc_engine.sdc.smart_defaults import calculate_smart_defaults
            return calculate_smart_defaults(
                df, detected_qis, initial_reid_95,
                structural_risk=structural_risk)
        except Exception as e:
            logging.error(f"Smart defaults calculation failed: {e}")
            return None

    def get_preprocessing_report(self) -> Dict[str, Any]:
        """Run quick pre-anonymization assessment and return report dict."""
        df = self.dataset.get()
        try:
            from sdc_engine.sdc.sdc_preprocessing import (
                generate_pre_anonymization_report,
            )
            return generate_pre_anonymization_report(df)
        except Exception as e:
            logging.error(f"Pre-anonymization report failed: {e}")
            return {}

    def apply_preprocessing(
        self,
        quasi_identifiers: List[str],
        mode: str = 'auto',
        config: Optional[Dict] = None,
        utility_threshold: float = 0.0,
        sensitive_columns: Optional[List[str]] = None,
    ) -> PreprocessingResult:
        """
        Apply preprocessing pipeline with utility threshold enforcement.

        Each optional step is applied incrementally. After each step,
        utility is measured. If utility drops below ``utility_threshold``,
        that step is rolled back and subsequent optional steps are skipped.

        Parameters
        ----------
        quasi_identifiers : list
            Column names selected as quasi-identifiers.
        mode : str
            'auto', 'mandatory_only', 'full', 'report_only', 'none'
        config : dict, optional
            Override default preprocessing rules.
        utility_threshold : float, default=0.0
            Minimum acceptable utility (0-1). Steps that push utility
            below this threshold are skipped and reported.
        sensitive_columns : list, optional
            Columns to measure utility on (analysis variables).
            When provided, utility is measured on these columns only
            (SDC framing: preserve what analysts need).
            When None, falls back to all non-QI columns.

        Returns
        -------
        PreprocessingResult
        """
        df = self.dataset.get()
        result = PreprocessingResult()
        result.original_data = df.copy()

        # Extract column_types from config for utility measurements
        _ct = (config or {}).get('column_types') or None

        try:
            from sdc_engine.sdc.sdc_preprocessing import preprocess_for_sdc
            from sdc_engine.sdc.sdc_utils import calculate_reid

            # ReID before
            try:
                reid_before = calculate_reid(df, quasi_identifiers)
                result.reid_before = reid_before.get('reid_95', 0)
            except Exception:
                result.reid_before = None

            # --- Step-by-step with utility checking ---
            if utility_threshold > 0 and config:
                # Run steps individually to enforce threshold
                current = df.copy()
                steps_order = [
                    ('remove_direct_identifiers', 'Remove direct IDs'),
                    ('top_bottom_coding', 'Top/bottom coding'),
                    ('merge_rare_categories', 'Merge rare categories'),
                    ('generalize', 'GENERALIZE'),
                ]
                progression = [{'step': 'Original', 'utility': 1.0,
                                'reid': result.reid_before or 0}]
                threshold = utility_threshold / 100.0  # Convert from % to 0-1

                logging.info(f"[PREPROCESS] Step-by-step mode: threshold={threshold:.0%}, "
                             f"QIs={quasi_identifiers}, sensitive={sensitive_columns}")

                for step_key, step_label in steps_order:
                    step_cfg = config.get(step_key, {})
                    if not step_cfg.get('enabled', False):
                        continue

                    # Build config with ONLY this step enabled
                    single_config = {k: {'enabled': False} for k in
                                     ['remove_direct_identifiers',
                                      'top_bottom_coding',
                                      'merge_rare_categories',
                                      'generalize']}
                    single_config[step_key] = dict(step_cfg)
                    # Pass column_types so each step uses Configure's
                    # classification as source of truth (no re-probing).
                    if config.get('column_types'):
                        single_config['column_types'] = config['column_types']
                    # Pass per-QI treatment levels for differentiated preprocessing
                    if config.get('qi_treatment'):
                        single_config['qi_treatment'] = config['qi_treatment']

                    # Inject per-QI utility gate into GENERALIZE step
                    if step_key == 'generalize':
                        _orig = df  # closure captures original data
                        _sens = sensitive_columns
                        _qis = quasi_identifiers

                        _col_types = _ct
                        def _utility_fn(candidate_df, qi_name,
                                        _o=_orig, _s=_sens, _q=_qis,
                                        _colt=_col_types):
                            try:
                                b, _, _, _ = SDCPreprocessing._compute_blended_utility(
                                    _o, candidate_df, _q, _s,
                                    column_types=_colt)
                            except Exception:
                                return 1.0
                            try:
                                # Fast proxy: subgroup mean preservation
                                # for this QI only (skips η², TVD, corr matrix)
                                sg = compute_fast_qi_utility(
                                    _o, candidate_df, qi_name, _s)
                                # Blend like composite: 30% per-var + 70% subgroup
                                return 0.30 * b + 0.70 * sg
                            except Exception:
                                return b

                        single_config[step_key]['utility_fn'] = _utility_fn
                        single_config[step_key]['utility_threshold'] = threshold

                    try:
                        candidate, meta = preprocess_for_sdc(
                            data=current,
                            quasi_identifiers=quasi_identifiers,
                            mode='full',
                            config=single_config,
                            return_metadata=True,
                        )
                    except Exception:
                        continue

                    # Was this step actually applied?
                    if step_key not in meta.get('steps_applied', []):
                        # Step found no work to do (e.g., no outliers)
                        result.steps_skipped.append({
                            'step': step_key,
                            'label': step_label,
                            'utility_after': None,
                            'threshold': threshold,
                            'reason': 'No applicable data found',
                        })
                        continue

                    # Full composite utility at each step: blended
                    # per-variable (sensitive + QI) then cross-tab.
                    import time as _time
                    _t0 = _time.perf_counter()

                    try:
                        blended, util, qi_avg, _pv = (
                            self._compute_blended_utility(
                                df, candidate,
                                quasi_identifiers, sensitive_columns,
                                column_types=_ct))
                    except Exception:
                        util, qi_avg, blended = 1.0, 1.0, 1.0

                    # Cross-tab benchmark + composite (same as final score)
                    gate_score = blended
                    try:
                        _bench = compute_benchmark_analysis(
                            df, candidate, quasi_identifiers,
                            sensitive_columns=sensitive_columns,
                            column_types=_ct)
                        gate_score = compute_composite_utility(
                            blended, _bench)
                    except Exception:
                        pass  # fall back to blended if benchmark fails

                    _elapsed = (_time.perf_counter() - _t0) * 1000
                    logging.info(
                        f"[PREPROCESS] {step_label}: "
                        f"sens={util:.1%}, qi_avg={qi_avg:.1%}, "
                        f"blended={blended:.1%}, "
                        f"gate(composite)={gate_score:.1%}, "
                        f"threshold={threshold:.0%}, "
                        f"took={_elapsed:.0f}ms")

                    # Mandatory step (direct IDs) always applied
                    is_mandatory = step_key == 'remove_direct_identifiers'

                    # GENERALIZE touches many QIs cumulatively — use a
                    # relaxed step-level threshold so the step isn't
                    # rejected when per-QI gates already accepted each QI.
                    step_threshold = (threshold * 0.8 if step_key == 'generalize'
                                      else threshold)

                    if is_mandatory or gate_score >= step_threshold:
                        current = candidate
                        result.steps_applied.append(step_key)
                        logging.info(f"[PREPROCESS] >> {step_label} ACCEPTED "
                                     f"(gate={gate_score:.1%} >= {step_threshold:.0%})")
                        # Capture GENERALIZE early-exit metadata
                        gen_meta = meta.get('changes_summary', {}).get(
                            'generalize', {})
                        if gen_meta.get('early_exit'):
                            result.generalize_early_exit = gen_meta
                        try:
                            reid_now = calculate_reid(current, quasi_identifiers)
                            reid_val = reid_now.get('reid_95', 0)
                        except Exception:
                            reid_val = 0
                        progression.append({
                            'step': step_label,
                            'utility': gate_score,
                            'sensitive_utility': util,
                            'qi_utility': qi_avg,
                            'reid': reid_val,
                        })
                    else:
                        logging.info(f"[PREPROCESS] >> {step_label} SKIPPED "
                                     f"(gate={gate_score:.1%} < {step_threshold:.0%})")
                        result.steps_skipped.append({
                            'step': step_key,
                            'label': step_label,
                            'utility_after': gate_score,
                            'threshold': step_threshold,
                            'reason': (
                                f'Utility would drop to {gate_score:.1%}, '
                                f'below threshold {step_threshold:.0%}'
                            ),
                        })

                result.preprocessed_data = current
                result.step_utility_progression = progression
                result.metadata = {'steps_applied': result.steps_applied}
            else:
                # No threshold — apply all at once (original behavior)
                preprocessed, metadata = preprocess_for_sdc(
                    data=df,
                    quasi_identifiers=quasi_identifiers,
                    mode=mode,
                    config=config,
                    return_metadata=True,
                )
                result.preprocessed_data = preprocessed
                result.metadata = metadata
                result.steps_applied = metadata.get('steps_applied', [])
                gen_meta = metadata.get('changes_summary', {}).get(
                    'generalize', {})
                if gen_meta.get('early_exit'):
                    result.generalize_early_exit = gen_meta
                result.step_details = metadata.get('step_details', [])

            # ReID after
            try:
                reid_after = calculate_reid(
                    result.preprocessed_data, quasi_identifiers)
                result.reid_after = reid_after.get('reid_95', 0)
            except Exception:
                result.reid_after = None

            # Blended utility: sensitive + QI per-variable
            sens_util = None
            qi_avg_final = 1.0
            try:
                blended, sens_util, qi_avg_final, per_var = (
                    self._compute_blended_utility(
                        df, result.preprocessed_data,
                        quasi_identifiers, sensitive_columns,
                        column_types=_ct))
                result.utility_score = blended
                result.per_variable_utility = per_var
            except Exception:
                result.utility_score = None

            # Comprehensive metrics on sensitive columns (primary)
            try:
                result.il1s = compute_il1s(
                    df, result.preprocessed_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=_ct)
                result.benchmark = compute_benchmark_analysis(
                    df, result.preprocessed_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=_ct)
                result.distributional = compute_distributional_metrics(
                    df, result.preprocessed_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=_ct)
            except Exception:
                pass
            try:
                from sdc_engine.sdc.metrics import calculate_utility_metrics
                result.comprehensive_metrics = calculate_utility_metrics(
                    df, result.preprocessed_data, quasi_identifiers)
            except Exception:
                pass

            # Composite utility: blend per-variable score with cross-tab
            # preservation so the displayed number reflects analytical
            # relationship integrity, not just per-variable identity.
            if result.utility_score is not None and result.benchmark:
                pre_composite = result.utility_score
                result.utility_score = compute_composite_utility(
                    pre_composite, result.benchmark,
                    per_variable_utility=result.per_variable_utility)
                logging.info(
                    f"[PREPROCESS] Final: sens={sens_util:.1%}, "
                    f"qi_avg={qi_avg_final:.1%}, "
                    f"blended={pre_composite:.1%}, "
                    f"composite={result.utility_score:.1%}, "
                    f"ReID={result.reid_after}")
            elif result.utility_score is not None:
                logging.info(
                    f"[PREPROCESS] Final: sens={sens_util:.1%}, "
                    f"qi_avg={qi_avg_final:.1%}, "
                    f"score={result.utility_score:.1%}, "
                    f"ReID={result.reid_after} (no cross-tab)")

            result.success = True

        except Exception as e:
            logging.error(f"Preprocessing failed: {e}")
            result.error = str(e)

        return result

    def apply_smart_workflow(
        self,
        quasi_identifiers: List[str],
        sensitive_columns: Optional[List[str]] = None,
    ) -> PreprocessingResult:
        """
        Run the full smart-defaults workflow (preprocess + protect in one shot).

        Returns a PreprocessingResult with the preprocessed data and metrics.
        """
        df = self.dataset.get()
        result = PreprocessingResult()
        result.original_data = df.copy()

        try:
            from sdc_engine.sdc.smart_defaults import apply_smart_workflow
            from sdc_engine.sdc.sdc_utils import calculate_reid

            # ReID before
            try:
                reid_before = calculate_reid(df, quasi_identifiers)
                result.reid_before = reid_before.get('reid_95', 0)
            except Exception:
                result.reid_before = None

            protected_data, workflow_result = apply_smart_workflow(
                data=df,
                detected_qis=quasi_identifiers,
                initial_reid_95=result.reid_before or 0.5,
            )

            result.preprocessed_data = protected_data
            result.smart_defaults = workflow_result.get('smart_defaults')
            result.metadata = workflow_result

            # Extract ReID after from workflow result
            result.reid_after = workflow_result.get('final_reid_95')
            if result.reid_after is None:
                try:
                    reid_after = calculate_reid(protected_data, quasi_identifiers)
                    result.reid_after = reid_after.get('reid_95', 0)
                except Exception:
                    pass

            result.steps_applied = ['smart_workflow']

            # Blended utility: sensitive + QI per-variable
            try:
                blended, _s, _q, per_var = self._compute_blended_utility(
                    df, protected_data, quasi_identifiers, sensitive_columns,
                    column_types=_ct)
                result.utility_score = blended
                result.per_variable_utility = per_var
            except Exception:
                pass

            # Comprehensive metrics + composite
            try:
                result.il1s = compute_il1s(
                    df, protected_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=_ct)
                result.benchmark = compute_benchmark_analysis(
                    df, protected_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=_ct)
                result.distributional = compute_distributional_metrics(
                    df, protected_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=_ct)
            except Exception:
                pass
            if result.utility_score is not None and result.benchmark:
                result.utility_score = compute_composite_utility(
                    result.utility_score, result.benchmark,
                    per_variable_utility=result.per_variable_utility)

            result.success = True

        except Exception as e:
            logging.error(f"Smart workflow failed: {e}")
            result.error = str(e)

        return result

    def apply_smart_workflow_adaptive(
        self,
        quasi_identifiers: List[str],
        target_reid: float = 0.05,
        start_tier: str = 'light',
        sensitive_columns: Optional[List[str]] = None,
        var_priority: Optional[Dict] = None,
        structural_risk: float = 0.0,
    ) -> PreprocessingResult:
        """
        Run the adaptive retry workflow that progressively increases
        preprocessing aggressiveness until target ReID is met.

        Tiers: light (max_cat=15) → moderate (10) → aggressive (5)
        → very_aggressive (3).

        Returns a PreprocessingResult with attempt tracking metadata.
        """
        df = self.dataset.get()
        result = PreprocessingResult()
        result.original_data = df.copy()

        try:
            from sdc_engine.sdc.smart_defaults import (
                apply_smart_workflow_with_adaptive_retry,
            )
            from sdc_engine.sdc.sdc_utils import calculate_reid

            # ReID before
            try:
                reid_before = calculate_reid(df, quasi_identifiers)
                result.reid_before = reid_before.get('reid_95', 0)
            except Exception:
                result.reid_before = None

            protected_data, workflow_result = (
                apply_smart_workflow_with_adaptive_retry(
                    data=df,
                    detected_qis=quasi_identifiers,
                    initial_reid_95=result.reid_before or 0.5,
                    target_reid=target_reid,
                    start_tier=start_tier,
                    var_priority=var_priority or None,
                    structural_risk=structural_risk,
                )
            )

            # Store preprocessed-only data for utility comparison
            # (protected_data includes protection method on top)
            preprocessed_only = workflow_result.get('preprocessed_data')
            result.preprocessed_data = preprocessed_only if preprocessed_only is not None else protected_data
            result.smart_defaults = workflow_result.get('smart_defaults')
            result.metadata = workflow_result

            # Extract ReID after preprocessing (before protection)
            result.reid_after = workflow_result.get('reid_after_preprocess')
            if result.reid_after is None:
                result.reid_after = workflow_result.get('final_reid_95')
            if result.reid_after is None:
                try:
                    reid_after = calculate_reid(
                        result.preprocessed_data, quasi_identifiers)
                    result.reid_after = reid_after.get('reid_95', 0)
                except Exception:
                    pass

            # Record which tier succeeded
            tier_label = workflow_result.get('tier_label', '?')
            result.steps_applied = [f'adaptive_retry ({tier_label})']

            # Blended utility: sensitive + QI per-variable
            compare_data = result.preprocessed_data
            try:
                blended, _s, _q, per_var = self._compute_blended_utility(
                    df, compare_data, quasi_identifiers, sensitive_columns,
                    column_types=_ct)
                result.utility_score = blended
                result.per_variable_utility = per_var
            except Exception:
                pass

            # Comprehensive metrics + composite
            try:
                result.il1s = compute_il1s(
                    df, compare_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=_ct)
                result.benchmark = compute_benchmark_analysis(
                    df, compare_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=_ct)
                result.distributional = compute_distributional_metrics(
                    df, compare_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=_ct)
            except Exception:
                pass
            if result.utility_score is not None and result.benchmark:
                result.utility_score = compute_composite_utility(
                    result.utility_score, result.benchmark,
                    per_variable_utility=result.per_variable_utility)

            result.success = True

        except Exception as e:
            logging.error(f"Adaptive workflow failed: {e}")
            result.error = str(e)

        return result

    # Utility functions delegated to sdc_engine.sdc.utility module
    # (kept as static aliases for backwards compatibility)
    _compute_utility = staticmethod(compute_utility)
    _compute_per_variable_utility = staticmethod(compute_per_variable_utility)

    @staticmethod
    def _compute_blended_utility(
        original: pd.DataFrame,
        processed: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_columns: Optional[List[str]] = None,
        column_types: Optional[Dict[str, str]] = None,
    ) -> tuple:
        """Compute utility blending sensitive + QI per-variable scores.

        Preprocessing only modifies QIs, so sensitive utility alone is
        always ~100% and uninformative.  This method blends QI per-variable
        utility into the score so the number reflects actual information
        loss.

        Returns (blended_score, sens_util, qi_avg, per_variable_dict).
        """
        sens_util = compute_utility(
            original, processed,
            sensitive_columns=sensitive_columns,
            quasi_identifiers=quasi_identifiers,
            column_types=column_types,
        )
        per_var = compute_per_variable_utility(
            original, processed, quasi_identifiers,
            column_types=column_types)
        qi_scores = []
        for m in per_var.values():
            s = (m.get('correlation') if m.get('dtype') == 'numeric'
                 else m.get('row_preservation',
                            m.get('category_overlap', 1.0)))
            if s is not None:
                qi_scores.append(abs(s))
        qi_avg = sum(qi_scores) / len(qi_scores) if qi_scores else 1.0

        # When sensitive untouched (preprocessing), QI drives the score.
        # When both affected (protection), blend 50/50.
        if sens_util >= 0.95:
            blended = qi_avg
        else:
            blended = 0.5 * sens_util + 0.5 * qi_avg

        return blended, sens_util, qi_avg, per_var
