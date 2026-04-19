"""
SDC Protection Interactor
=========================

Orchestrates SDC method execution on a dataset.
Bridges the data model with SDC's protection methods.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import logging

from sdc_engine.entities.dataset.base import BaseDataset

log = logging.getLogger(__name__)


@dataclass
class ProtectionResult:
    """Result of applying an SDC protection method."""
    method: str = ''
    protected_data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reid_before: Optional[Dict[str, float]] = None
    reid_after: Optional[Dict[str, float]] = None
    utility_metrics: Optional[Dict[str, float]] = None
    # Unified metrics from sdc/utility.py (same as PreprocessingResult)
    utility_score: Optional[float] = None
    per_variable_utility: Optional[Dict] = None
    il1s: Optional[Dict] = None
    benchmark: Optional[Dict] = None
    distributional: Optional[Dict] = None
    success: bool = False
    error: str = ''
    # QI over-suppression flag: set when a structural method suppressed >40% of any QI
    qi_over_suppressed: bool = False
    qi_suppression_detail: Optional[Dict[str, float]] = None
    # Pluggable risk metric support
    risk_assessment_before: Any = None   # RiskAssessment (before protection)
    risk_assessment_after: Any = None    # RiskAssessment (after protection)


@dataclass
class ComparisonResult:
    """Result of comparing multiple SDC methods."""
    results: List[ProtectionResult] = field(default_factory=list)
    summary: Optional[pd.DataFrame] = None
    best_method: str = ''


# Available microdata methods and their display info
MICRODATA_METHODS = {
    'kANON': {'name': 'k-Anonymity', 'description': 'Generalization + suppression to achieve k-anonymity'},
    'PRAM': {'name': 'PRAM', 'description': 'Post-Randomization (category swapping)'},
    'NOISE': {'name': 'Noise Addition', 'description': 'Add random noise to continuous variables'},
    'LOCSUPR': {'name': 'Local Suppression', 'description': 'Targeted cell suppression for k-anonymity'},
}

TABULAR_METHODS = {}  # Tabular methods removed; microdata methods cover all use cases


@dataclass
class SDCProtection:
    """Interactor for running SDC protection methods."""

    dataset: BaseDataset

    def get_available_methods(self, data_type: str = 'microdata') -> Dict[str, Dict]:
        """Return available methods based on data type."""
        if data_type == 'tabular':
            logging.info(
                "Tabular methods not available — microdata methods will be used.")
        return MICRODATA_METHODS

    def apply_method(
        self,
        method: str,
        quasi_identifiers: List[str],
        params: Optional[Dict[str, Any]] = None,
        use_active_cols: bool = True,
        input_data: Optional[pd.DataFrame] = None,
        sensitive_columns: Optional[List[str]] = None,
        column_types: Optional[Dict[str, str]] = None,
        qi_treatment: Optional[Dict[str, str]] = None,
        l_target: Optional[int] = None,
        t_target: Optional[float] = None,
        use_r: bool = True,
    ) -> ProtectionResult:
        """Apply a single SDC method to the dataset.

        Parameters
        ----------
        input_data : DataFrame, optional
            If provided, use this instead of self.dataset.get().
            Useful for applying protection to preprocessed data.
        sensitive_columns : list, optional
            Columns to measure utility on (analysis variables).
            When provided, utility is measured on these columns only.
            When None, falls back to all non-QI columns.
        qi_treatment : dict, optional
            Per-QI treatment levels ``{col: 'Heavy'|'Standard'|'Light'}``.
            When provided, methods receive per-variable parameter dicts
            scaled by treatment multipliers (Heavy=1.5×, Light=0.5×).
        use_r : bool
            When True, LOCSUPR and NOISE attempt R/sdcMicro backend.
        """
        result = ProtectionResult(method=method)
        log.info("[Interactor] apply_method: %s  qis=%s  params=%s",
                 method, quasi_identifiers, {k: v for k, v in (params or {}).items()
                                              if not k.startswith('_')})

        if input_data is not None:
            df = input_data
        else:
            df = self.dataset.get()
            if use_active_cols:
                active_cols = self.dataset.get_active_columns()
                df = df[active_cols] if active_cols else df

        if params is None:
            params = {}

        # Calculate ReID before
        try:
            from sdc_engine.sdc.sdc_utils import calculate_reid
            result.reid_before = calculate_reid(df, quasi_identifiers)
        except Exception:
            pass

        # Apply the method
        try:
            method_func = self._get_method_func(method)
            if method_func is None:
                result.error = f"Unknown method: {method}"
                return result

            # Build method-specific kwargs
            method_kwargs = self._build_method_kwargs(
                method, quasi_identifiers, params,
                column_types=column_types,
                qi_treatment=qi_treatment,
                sensitive_columns=sensitive_columns,
                l_target=l_target,
                t_target=t_target,
                use_r=use_r,
            )

            output = method_func(
                df.copy(),
                return_metadata=True,
                verbose=False,
                **method_kwargs,
            )

            if isinstance(output, tuple):
                result.protected_data, result.metadata = output
            else:
                result.protected_data = output

            # Two-pass: apply lighter perturbation to sensitive columns
            sens_cols = params.get('_protect_sensitive_cols') if params else None
            if sens_cols and method.upper() in ('NOISE', 'PRAM'):
                sens_kwargs = {'variables': sens_cols}
                if method.upper() == 'NOISE':
                    sens_kwargs['magnitude'] = params.get(
                        '_sensitive_magnitude', 0.05)
                elif method.upper() == 'PRAM':
                    sens_kwargs['p_change'] = params.get(
                        '_sensitive_p_change', 0.10)
                sens_output = method_func(
                    result.protected_data,
                    return_metadata=True,
                    verbose=False,
                    **sens_kwargs,
                )
                if isinstance(sens_output, tuple):
                    result.protected_data, sens_meta = sens_output
                    result.metadata['sensitive_pass'] = sens_meta
                else:
                    result.protected_data = sens_output

            result.success = True
            log.info("[Interactor] %s succeeded: rows=%d",
                     method, len(result.protected_data) if result.protected_data is not None else 0)

            # ── QI over-suppression check ──────────────────────────────
            # If a structural method (kANON/LOCSUPR) suppressed >40% of
            # any QI column, flag the result so callers can try a fallback.
            if method.upper() in ('KANON', 'LOCSUPR') and quasi_identifiers:
                _qi_supp = {}
                for qi in quasi_identifiers:
                    if qi in df.columns and qi in result.protected_data.columns:
                        orig_valid = int(df[qi].notna().sum())
                        prot_valid = int(result.protected_data[qi].notna().sum())
                        if orig_valid > 0:
                            lost_pct = (orig_valid - prot_valid) / orig_valid
                            if lost_pct > 0.01:
                                _qi_supp[qi] = round(lost_pct, 3)
                if _qi_supp:
                    worst_qi = max(_qi_supp, key=_qi_supp.get)
                    worst_pct = _qi_supp[worst_qi]
                    if result.metadata and isinstance(result.metadata, dict):
                        result.metadata['qi_suppression_pct'] = _qi_supp
                    if worst_pct > 0.40:
                        log.warning(
                            "[Interactor] %s over-suppressed QI '%s' (%.0f%%) — "
                            "consider fallback to perturbation method",
                            method, worst_qi, worst_pct * 100)
                        result.qi_over_suppressed = True
                        result.qi_suppression_detail = _qi_supp

            # Calculate ReID after
            try:
                from sdc_engine.sdc.sdc_utils import calculate_reid
                result.reid_after = calculate_reid(result.protected_data, quasi_identifiers)
                log.info("[Interactor] %s ReID: before=%.4f  after=%.4f",
                         method,
                         result.reid_before.get('reid_95', -1) if result.reid_before else -1,
                         result.reid_after.get('reid_95', -1) if result.reid_after else -1)
            except Exception:
                pass

            # Calculate utility (weighted 6-component score)
            try:
                from sdc_engine.sdc.metrics import calculate_utility_metrics
                result.utility_metrics = calculate_utility_metrics(
                    df, result.protected_data, quasi_identifiers
                )
            except (ImportError, Exception):
                pass

            # Unified metrics (same as Preprocess — scoped to sensitive cols)
            try:
                from sdc_engine.sdc.utility import (
                    compute_utility, compute_per_variable_utility,
                    compute_il1s, compute_benchmark_analysis,
                    compute_distributional_metrics,
                    compute_composite_utility,
                )
                result.utility_score = compute_utility(
                    df, result.protected_data,
                    sensitive_columns=sensitive_columns,
                    quasi_identifiers=quasi_identifiers,
                    column_types=column_types,
                )
                result.per_variable_utility = compute_per_variable_utility(
                    df, result.protected_data, quasi_identifiers,
                    column_types=column_types)
                result.il1s = compute_il1s(
                    df, result.protected_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=column_types)
                result.benchmark = compute_benchmark_analysis(
                    df, result.protected_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=column_types)
                result.distributional = compute_distributional_metrics(
                    df, result.protected_data, quasi_identifiers,
                    sensitive_columns=sensitive_columns,
                    column_types=column_types)
                # Composite utility (per-variable on sensitive columns)
                if result.utility_score is not None and result.benchmark:
                    result.utility_score = compute_composite_utility(
                        result.utility_score, result.benchmark,
                        per_variable_utility=result.per_variable_utility)
            except Exception:
                pass

        except Exception as e:
            result.error = str(e)
            log.error("[Interactor] SDC method %s failed: %s", method, e, exc_info=True)

        return result

    def compare_methods(
        self,
        methods: List[str],
        quasi_identifiers: List[str],
        method_params: Optional[Dict[str, Dict]] = None,
    ) -> ComparisonResult:
        """Run multiple SDC methods and compare results."""
        comparison = ComparisonResult()

        if method_params is None:
            method_params = {}

        for method in methods:
            params = method_params.get(method, {})
            result = self.apply_method(method, quasi_identifiers, params)
            comparison.results.append(result)

        # Build summary table
        rows = []
        for r in comparison.results:
            row = {
                'Method': r.method,
                'Success': r.success,
                'ReID95 Before': r.reid_before.get('reid_95', None) if r.reid_before else None,
                'ReID95 After': r.reid_after.get('reid_95', None) if r.reid_after else None,
                'Error': r.error if not r.success else '',
            }
            if r.utility_metrics:
                row['Utility Score'] = r.utility_metrics.get('overall_utility', None)
                row['Info Loss'] = r.utility_metrics.get('information_loss', None)
            rows.append(row)

        comparison.summary = pd.DataFrame(rows)

        # Find best method (lowest ReID95 After among successes)
        successes = [r for r in comparison.results if r.success and r.reid_after]
        if successes:
            best = min(successes, key=lambda r: r.reid_after.get('reid_95', float('inf')))
            comparison.best_method = best.method

        return comparison

    def smart_protect(
        self,
        quasi_identifiers: List[str],
        target_reid: float = 0.05,
    ) -> ProtectionResult:
        """Auto-select and apply the best method based on ReID analysis."""
        df = self.dataset.get()
        active_cols = self.dataset.get_active_columns()
        df_active = df[active_cols] if active_cols else df

        try:
            from sdc_engine.sdc.select_method import recommend_method
            rec = recommend_method(df_active, quasi_identifiers=quasi_identifiers)

            primary_method = rec.get('primary', 'kANON')
            params = rec.get('params', {})

            return self.apply_method(primary_method, quasi_identifiers, params)
        except Exception as e:
            logging.error(f"Smart protection failed: {e}")
            # Fallback to kANON
            return self.apply_method('kANON', quasi_identifiers, {'k': 5})

    @staticmethod
    def _build_method_kwargs(
        method: str,
        quasi_identifiers: List[str],
        params: Dict[str, Any],
        column_types: Optional[Dict[str, str]] = None,
        qi_treatment: Optional[Dict[str, str]] = None,
        sensitive_columns: Optional[List[str]] = None,
        l_target: Optional[int] = None,
        t_target: Optional[float] = None,
        use_r: bool = True,
    ) -> Dict[str, Any]:
        """Build method-specific keyword arguments.

        Different SDC methods accept different parameter names
        (e.g., NOISE uses 'variables' instead of 'quasi_identifiers').

        When *qi_treatment* is provided, per-variable parameter dicts
        are built via the centralized ``qi_treatment`` module so that
        each QI receives Heavy/Standard/Light scaled parameters.
        """
        kwargs = dict(params)  # copy user params

        # Methods that take quasi_identifiers
        qi_methods = {'KANON', 'LOCSUPR'}
        # Methods that take 'variables' for column selection
        var_methods = {'NOISE', 'PRAM'}

        upper = method.upper()
        if upper in qi_methods:
            kwargs.setdefault('quasi_identifiers', quasi_identifiers)
        elif upper in var_methods:
            # Pop sensitive-pass keys (handled separately in apply_method)
            kwargs.pop('_protect_sensitive_cols', None)
            kwargs.pop('_sensitive_magnitude', None)
            kwargs.pop('_sensitive_p_change', None)
            kwargs.setdefault('variables', quasi_identifiers)

        # Pass column_types to methods that support it
        if column_types:
            kwargs.setdefault('column_types', column_types)

        # ── Per-QI treatment scaling ──────────────────────────────
        if qi_treatment:
            from sdc_engine.sdc.qi_treatment import (
                build_per_variable_params,
                build_locsupr_weights,
                get_method_base_defaults,
            )
            defaults = get_method_base_defaults()

            if upper == 'NOISE':
                base_mag = kwargs.get('magnitude', defaults.get('NOISE', 0.10))
                pv = build_per_variable_params(
                    'NOISE', base_mag, quasi_identifiers, qi_treatment)
                if pv:
                    kwargs['per_variable_magnitude'] = pv

            elif upper == 'PRAM':
                base_pc = kwargs.get('p_change', defaults.get('PRAM', 0.20))
                pv = build_per_variable_params(
                    'PRAM', base_pc, quasi_identifiers, qi_treatment)
                if pv:
                    kwargs['per_variable_p_change'] = pv

            elif upper == 'KANON':
                base_bs = kwargs.get('bin_size', defaults.get('KANON', 10))
                pv = build_per_variable_params(
                    'KANON', base_bs, quasi_identifiers, qi_treatment)
                if pv:
                    kwargs['per_qi_bin_size'] = pv

        # Pass l-diversity / t-closeness targets and sensitive columns to kANON
        if upper == 'KANON':
            if sensitive_columns:
                kwargs.setdefault('sensitive_columns', sensitive_columns)
            if l_target is not None:
                kwargs.setdefault('l_target', l_target)
            if t_target is not None:
                kwargs.setdefault('t_target', t_target)

        elif upper == 'LOCSUPR':
            # Only inject if user hasn't already set importance_weights
            if qi_treatment and 'importance_weights' not in kwargs:
                from sdc_engine.sdc.qi_treatment import build_locsupr_weights
                weights = build_locsupr_weights(
                    quasi_identifiers, qi_treatment)
                if weights:
                    kwargs['importance_weights'] = weights

        # Pass use_r preference to methods that support R/sdcMicro backend
        if upper in ('NOISE', 'LOCSUPR'):
            kwargs['use_r'] = use_r

        return kwargs

    @staticmethod
    def _get_method_func(method: str):
        """Get the SDC method function by name."""
        # Use uppercase keys for case-insensitive lookup
        method_map = {
            'KANON': 'apply_kanon',
            'PRAM': 'apply_pram',
            'NOISE': 'apply_noise',
            'LOCSUPR': 'apply_locsupr',
        }
        func_name = method_map.get(method.upper())
        if func_name is None:
            return None

        import importlib
        sdc = importlib.import_module('sdc_engine.sdc')
        return getattr(sdc, func_name, None)
