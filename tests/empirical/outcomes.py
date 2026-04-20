"""
Outcome scoring -- run a dataset under a specific threshold value
and return comparable metrics.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import re
import time
import pandas as pd


@dataclass
class Outcome:
    dataset: str
    threshold_id: str
    threshold_value: float
    risk_metric: str
    risk_target: float
    selected_method: str
    selected_rule: str
    initial_method: str          # what the rule selected BEFORE override
    reid_before: float
    reid_after: float
    min_k_before: Optional[float] = None
    min_k_after: Optional[float] = None
    utility_score: float = 0.0
    suppression_rate: float = 0.0
    n_iterations: int = 0
    elapsed_sec: float = 0.0
    target_met: bool = False
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_RULE_RE = re.compile(r"Rule:\s+(\S+)\s+→\s+(\S+)")


def _extract_rule_from_log(log_entries: List[str]) -> tuple:
    """Extract (rule_name, initial_method) from log entries.

    The log contains lines like: 'Rule: MED1_Moderate_Structural → kANON'
    """
    for entry in log_entries:
        m = _RULE_RE.search(entry)
        if m:
            return m.group(1), m.group(2)
    return '', ''


def run_outcome(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_columns: List[str],
    threshold_id: str,
    threshold_value: float,
    patcher_ctx,
    risk_metric: str = 'reid95',
    risk_target: float = 0.05,
) -> Outcome:
    """Run protection on df under the patched threshold and return an Outcome."""
    start = time.monotonic()
    err: Optional[str] = None
    try:
        with patcher_ctx(threshold_value):
            # Force Python-only fallback: deterministic, no R/rpy2 variance
            from sdc_engine.sdc import r_backend as _rb
            _rb._R_CHECK_CACHE["result"] = False

            from sdc_engine.entities.dataset.pandas.dataset import PdDataset
            from sdc_engine.interactors.sdc_protection import SDCProtection
            from sdc_engine.sdc.protection_engine import (
                run_rules_engine_protection, build_data_features,
            )

            dataset = PdDataset(data=df.copy(), activeCols=list(df.columns))
            protector = SDCProtection(dataset=dataset)

            features = build_data_features(df, quasi_identifiers)
            features['_risk_metric_type'] = risk_metric
            features['_reid_target_raw'] = risk_target
            result, log_entries = run_rules_engine_protection(
                input_data=df,
                quasi_identifiers=quasi_identifiers,
                data_features=features,
                access_tier='SCIENTIFIC',
                reid_target=risk_target,
                utility_floor=0.80,
                apply_method_fn=protector.apply_method,
                sensitive_columns=sensitive_columns,
                risk_metric=risk_metric,
                risk_target_raw=risk_target,
            )

            reid_before = (result.reid_before or {}).get('reid_95', float('nan'))
            reid_after = (result.reid_after or {}).get('reid_95', float('nan'))

            # Extract k-anonymity metrics from the synthetic reid dict
            min_k_before = (result.reid_before or {}).get('_raw_value')
            min_k_after = (result.reid_after or {}).get('_raw_value')

            # If k_anonymity metric but _raw_value not in dict, compute directly
            if risk_metric == 'k_anonymity':
                from sdc_engine.sdc.metrics.risk import check_kanonymity
                try:
                    if min_k_before is None:
                        _, gs_b, _ = check_kanonymity(df, quasi_identifiers, k=1)
                        if len(gs_b) > 0:
                            # Size column is last column (renamed to 'count' or '_group_size_')
                            min_k_before = int(gs_b.iloc[:, -1].min())
                except Exception:
                    pass

                if min_k_after is None:
                    try:
                        protected = getattr(result, 'protected_data', None)
                        if protected is not None and len(protected) > 0:
                            _, gs, _ = check_kanonymity(protected, quasi_identifiers, k=1)
                            if len(gs) > 0:
                                min_k_after = int(gs.iloc[:, -1].min())
                    except Exception:
                        pass

            supp_detail = getattr(result, 'qi_suppression_detail', {}) or {}
            max_supp = max(supp_detail.values()) if supp_detail else 0

            rule_name, initial_method = _extract_rule_from_log(log_entries)

            # Metric-specific target_met check
            if risk_metric == 'k_anonymity':
                target_met = (min_k_after or 0) >= risk_target
            elif risk_metric == 'uniqueness':
                target_met = (reid_after or 1.0) <= risk_target
            else:  # reid95
                target_met = (reid_after or 1.0) <= risk_target

            return Outcome(
                dataset="",  # filled by caller
                threshold_id=threshold_id,
                threshold_value=threshold_value,
                risk_metric=risk_metric,
                risk_target=risk_target,
                selected_method=result.method or "UNKNOWN",
                selected_rule=rule_name,
                initial_method=initial_method,
                reid_before=reid_before,
                reid_after=reid_after,
                min_k_before=min_k_before,
                min_k_after=min_k_after,
                utility_score=result.utility_score or 0,
                suppression_rate=max_supp,
                n_iterations=(result.metadata or {}).get('n_iterations', 0)
                             if result.metadata else 0,
                elapsed_sec=time.monotonic() - start,
                target_met=target_met,
            )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        return Outcome(
            dataset="", threshold_id=threshold_id,
            threshold_value=threshold_value,
            risk_metric=risk_metric,
            risk_target=risk_target,
            selected_method="ERROR", selected_rule="",
            initial_method="",
            reid_before=float('nan'), reid_after=float('nan'),
            min_k_before=None, min_k_after=None,
            utility_score=0, suppression_rate=0,
            n_iterations=0, elapsed_sec=time.monotonic() - start,
            target_met=False, error=err,
        )
