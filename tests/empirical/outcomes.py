"""
Outcome scoring -- run a dataset under a specific threshold value
and return comparable metrics.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import time
import pandas as pd


@dataclass
class Outcome:
    dataset: str
    threshold_id: str
    threshold_value: float
    selected_method: str
    selected_rule: str
    reid_before: float
    reid_after: float
    utility_score: float
    suppression_rate: float
    n_iterations: int
    elapsed_sec: float
    target_met: bool
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_outcome(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_columns: List[str],
    threshold_id: str,
    threshold_value: float,
    patcher_ctx,
    risk_target: float = 0.05,
) -> Outcome:
    """Run protection on df under the patched threshold and return an Outcome."""
    start = time.monotonic()
    err: Optional[str] = None
    try:
        with patcher_ctx(threshold_value):
            from sdc_engine.entities.dataset.pandas.dataset import PdDataset
            from sdc_engine.interactors.sdc_protection import SDCProtection
            from sdc_engine.sdc.protection_engine import (
                run_rules_engine_protection, build_data_features,
            )

            dataset = PdDataset(data=df.copy(), activeCols=list(df.columns))
            protector = SDCProtection(dataset=dataset)

            features = build_data_features(df, quasi_identifiers)
            result, _log = run_rules_engine_protection(
                input_data=df,
                quasi_identifiers=quasi_identifiers,
                data_features=features,
                access_tier='SCIENTIFIC',
                reid_target=risk_target,
                utility_floor=0.80,
                apply_method_fn=protector.apply_method,
                sensitive_columns=sensitive_columns,
                risk_target_raw=risk_target,
            )

            reid_before = (result.reid_before or {}).get('reid_95', float('nan'))
            reid_after = (result.reid_after or {}).get('reid_95', float('nan'))
            supp_detail = getattr(result, 'qi_suppression_detail', {}) or {}
            max_supp = max(supp_detail.values()) if supp_detail else 0

            return Outcome(
                dataset="",  # filled by caller
                threshold_id=threshold_id,
                threshold_value=threshold_value,
                selected_method=result.method or "UNKNOWN",
                selected_rule=(result.metadata or {}).get('rule_applied', '')
                              if result.metadata else '',
                reid_before=reid_before,
                reid_after=reid_after,
                utility_score=result.utility_score or 0,
                suppression_rate=max_supp,
                n_iterations=(result.metadata or {}).get('n_iterations', 0)
                             if result.metadata else 0,
                elapsed_sec=time.monotonic() - start,
                target_met=bool(result.success),
            )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        return Outcome(
            dataset="", threshold_id=threshold_id,
            threshold_value=threshold_value,
            selected_method="ERROR", selected_rule="",
            reid_before=float('nan'), reid_after=float('nan'),
            utility_score=0, suppression_rate=0,
            n_iterations=0, elapsed_sec=time.monotonic() - start,
            target_met=False, error=err,
        )
