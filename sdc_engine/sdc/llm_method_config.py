"""
LLM-assisted preprocessing and protection method selection for SDC pipeline.

Builds a dataset profile, calls the Cerebras LLM for method recommendation,
translates the response to an internal AIConfig format, and provides a
validated merge function (apply_ai_config) that safely merges AI suggestions
into the rule-based engine config.

The AI can never reduce protection below the rules baseline.
"""

import copy
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from .llm_assistant import CerebrasAssistant, get_assistant

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — generic SDC method selection expertise
# Based on test_cerebras_method.ps1, made dataset-agnostic with
# aggressive preprocessing guidance and combination space math
# ---------------------------------------------------------------------------
_METHOD_SYSTEM_PROMPT = """\
You are an expert in Statistical Disclosure Control (SDC). You will REVIEW the rules engine's recommendation and either agree or propose an alternative.

REVIEW MODE:
You will receive a rules engine recommendation (method, k, rule name, reasoning).
1. Evaluate whether the rules engine's choice is correct for this specific dataset.
2. If you AGREE: use the same method and parameters, and explain why the rules engine's reasoning is sound.
3. If you DISAGREE: recommend an alternative method with specific reasoning about what the rules engine missed — e.g., it ignored suppression estimates, chose kANON on a small dataset, or picked PRAM for numeric-dominant QIs.
4. Your "reasoning" field MUST reference the rules engine recommendation explicitly (e.g., "The rules engine selected kANON k=7 via QR3, but estimated suppression at k=7 is 32% — switching to PRAM preserves all records").
5. If no rules engine recommendation is present, make an independent recommendation.

You will also recommend preprocessing steps AND a protection method for the dataset described in the user message.

FEASIBILITY MATH (you MUST do this):
1. Calculate the target combination space: n_records / target_k (e.g., 1000/5 = 200 max groups)
2. For each QI, estimate cardinality AFTER your proposed preprocessing
3. Multiply all post-preprocessing cardinalities = estimated combination space
4. If estimated combo space > target, your preprocessing is NOT sufficient. Adjust.
5. Rule of thumb: aim for combo space = target / 2 (gives room for uneven distributions)

PREPROCESSING AGGRESSIVENESS — scale to risk level:

LOW RISK (ReID95 < 20%, structural_risk < 30%):
- Light touch. Skip preprocessing for QIs under 100 unique values.
- Only preprocess QIs with >500 unique values.
- Prefer gentle actions: merge_rare over top_k, month over quarter.
- Target: reduce combination space enough for k=3-5, no more.
- NEVER apply aggressive preprocessing when ReID95 < 20%.

MODERATE RISK (ReID95 20-50%, structural_risk 30-50%):
- Standard preprocessing. Top-K for high-cardinality categoricals.
- Date truncation to quarter or month depending on cardinality.
- Target: combination space that supports k=5 with <15% suppression.

HIGH RISK (ReID95 > 50%, structural_risk > 50%):
- Aggressive preprocessing. Top-K with low K values.
- Date truncation to quarter or year.
- Target: combination space that supports k=5-7 with <10% suppression.

PREPROCESSING RULES:
1. Date truncation -- bin dates to year/quarter/month based on risk level.
2. Top-K generalization -- keep top N categories, merge rest to "Other":
   - >1000 unique: top-30 to top-50 (high risk) or top-100 (low risk)
   - 100-1000 unique: top-50 to top-100
   - <100 unique: skip or light merge
   NEVER use merge_rare alone on high-cardinality categoricals (>200 unique).
3. Geographic codes (zip/postal): truncate digits based on dataset size and risk.
4. Numeric binning -- decade bins for years, equal-width for other numerics.
5. Top/bottom coding -- cap outliers before binning skewed data.
6. Skip -- when cardinality is already manageable (<50 unique for high risk, <100 for low risk).
7. Hierarchy dedup -- if two QIs form a geographic/categorical hierarchy (municipality nests within prefecture, district within region), recommend "skip" for the finer-grained column. The classification step should have already dropped it, but if both appear as QIs, flag it and suggest skipping the finest to avoid combination space explosion.

PER-QI CARDINALITY TARGETS (scale to dataset size):
- For n_records < 2000: each QI should have ≤ 10 unique values after preprocessing, ideally ≤ 5
- For n_records 2000-10000: each QI should have ≤ 20 unique values after preprocessing
- For n_records > 10000: each QI should have ≤ 50 unique values after preprocessing
- Gender/binary columns (2-3 values): always skip
- NEVER leave a QI with >50 unique values after preprocessing regardless of dataset size

AVAILABLE PROTECTION METHODS (applied after preprocessing):
- kANON: k-anonymity via generalization + suppression. Structural guarantee. Best for: mixed QI types + high risk where suppression stays <15%.
- LOCSUPR: Local suppression -- blanks specific cells in risky records. Targeted. Best when: risk concentrated in 1-2 QIs, or as fallback when kANON suppression exceeds 15%.
- PRAM: Post-randomization -- perturbs categorical values. Preserves ALL records (zero suppression). Best for: all-categorical QIs at low-moderate risk. NOT effective when one category dominates >80%.
- NOISE: Noise addition -- adds calibrated random noise. Preserves distributions. Best for: numeric-dominant QIs, or when suppression must be avoided. NOT effective on categorical data.

METHOD SELECTION — choose based on actual data, NOT defaults:
- ALL categorical QIs + low risk (<20%) → PRAM (preserves all records, no suppression)
- ALL categorical QIs + moderate risk (20-40%) → PRAM with higher p_change, or kANON if suppression estimate is low
- Mostly numeric QIs + low-moderate risk → NOISE (preserves distributions)
- Mixed QI types + high risk → kANON (strongest guarantee)
- Risk concentrated in 1-2 QIs → LOCSUPR (targeted suppression)
- Estimated suppression > 15% after preprocessing → prefer PRAM/NOISE over kANON
- Small dataset (<5000 rows) → prefer PRAM/NOISE (structural methods cause excessive suppression)
- Large dataset (>10K) + high risk + mixed types → kANON is viable
NEVER recommend kANON with k>5 unless ReID95 > 40% after preprocessing.

PROTECTION CONTEXT influence:
- Public release (k>=10): structural methods preferred for provable guarantee
- Scientific use (k>=5): balanced — either structural or perturbative
- Internal (k>=3): perturbative preferred — minimize distortion

CONFIDENCE CALIBRATION:
- high: combo space math works out, estimated suppression <10%, method matches data characteristics
- medium: combo space is borderline, may need 10-20% suppression or fallback method
- low: combo space far exceeds target even after preprocessing, uncertain outcome

IMPORTANCE VECTOR for kANON (if recommended):
Assign per-QI importance 1-5 based on cardinality contribution:
- QIs with highest post-preprocessing cardinality get importance 5 (generalize these first)
- QIs already at low cardinality (2-5 values) get importance 1 (preserve these)
- Geographic hierarchy: finest level gets highest importance

ATTRIBUTE DISCLOSURE:
- Check if equivalence classes will have diverse sensitive values (l-diversity)
- Dual-role sensitive columns with high cardinality usually have natural diversity -- flag but do not perturb by default

REASONING — justify against alternatives:
In the "reasoning" field, explain why the chosen method is better than at least one alternative for THIS specific dataset. Not generic "kANON provides guarantees" but specific reasoning referencing QI types, risk level, combo space, and estimated suppression.

Respond ONLY in JSON format:
{
  "preprocessing": {
    "plan": [
      {
        "qi": "column name",
        "action": "date_truncation|top_k|numeric_binning|top_bottom_coding|geographic_truncation|merge_rare|skip",
        "params": {"target_cardinality": N},
        "reasoning": "1 sentence why"
      }
    ],
    "expected_cardinality_after": {"qi_name": estimated_unique_after},
    "estimated_combination_space_after": N,
    "target_combination_space": N,
    "feasibility": "feasible|borderline|infeasible"
  },
  "rules_review": {
    "agrees": true or false,
    "rules_method": "what the rules engine recommended",
    "override_reason": "why you disagree (omit if agrees=true)"
  },
  "protection": {
    "method": "kANON|LOCSUPR|PRAM|NOISE",
    "params": {},
    "reasoning": "2-3 sentences referencing the combo space math AND the rules engine recommendation",
    "estimated_suppression": "e.g. ~8%",
    "confidence": "high|medium|low",
    "importance_vector": {"qi_name": 1-5},
    "alternative": {
      "method": "...",
      "params": {},
      "when": "use this if primary fails or suppression exceeds X%"
    }
  },
  "attribute_disclosure": {
    "risk_level": "low|moderate|high",
    "reasoning": "1-2 sentences",
    "suggestion": "none|optional NOISE on [columns]"
  },
  "expected_outcome": {
    "estimated_reid_after": "e.g. <5%",
    "estimated_utility": "e.g. ~80%",
    "estimated_k_achieved": 5,
    "key_tradeoff": "1 sentence summary"
  },
  "warnings": ["any concerns or caveats"]
}

No markdown, no preamble, no explanation outside the JSON."""


# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------
VALID_METHODS = {"kANON", "LOCSUPR", "PRAM", "NOISE"}
VALID_TREATMENTS = {"Heavy", "Standard", "Light"}
VALID_ACTIONS = {
    "skip", "generalize", "date_truncate", "age_bin",
    "geo_coarsen", "top_bottom", "numeric_round",
    # Also accept LLM action names from the prompt
    "date_truncation", "top_k", "numeric_binning",
    "top_bottom_coding", "merge_rare",
}

# Parameter clamp ranges: (min, max)
PARAM_RANGES = {
    "k": (2, 50),
    "p_change": (0.05, 0.50),
    "magnitude": (0.01, 0.50),
    "max_categories": (3, 50),
    "max_suppression_rate": (0.01, 0.30),
    "bin_size": (2, 50),
}


def _match_method(name: str) -> Optional[str]:
    """Case-insensitive match against VALID_METHODS. Returns canonical name or None."""
    if not name:
        return None
    return next((m for m in VALID_METHODS if m.upper() == name.upper()), None)


def _clamp(value: float, param_name: str) -> float:
    """Clamp a parameter value to its valid range."""
    if param_name in PARAM_RANGES:
        lo, hi = PARAM_RANGES[param_name]
        return max(lo, min(hi, value))
    return value


def _json_default(obj: Any) -> Any:
    """Custom JSON serializer for numpy/pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    return str(obj)


# ---------------------------------------------------------------------------
# Dataset profile builder
# ---------------------------------------------------------------------------
def _build_dataset_profile(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_columns: List[str],
    risk_metrics: Dict[str, float],
    qi_treatment: Optional[Dict[str, str]] = None,
    protection_context: str = "scientific_use",
    dataset_description: Optional[str] = None,
) -> str:
    """Build the user prompt for method selection from dataset profile.

    Parameters
    ----------
    dataset_description : str, optional
        User-provided description like "hospital patient records studying
        readmission rates". When provided, the LLM can make better
        domain-specific decisions about method selection and preprocessing
        aggressiveness.
    """
    qi_treatment = qi_treatment or {}
    n_records = len(data)

    if dataset_description:
        header = f"Dataset: {dataset_description}, {n_records:,} records, {len(data.columns)} columns"
    else:
        header = f"Dataset: {n_records:,} records, {len(data.columns)} columns"

    lines = [
        header,
        f"Protection context: {protection_context} "
        f"(target k>={risk_metrics.get('target_k', 5)}, "
        f"ReID95<={risk_metrics.get('target_reid', 5)}%, "
        f"utility floor {risk_metrics.get('utility_floor', 90)}%)",
        "",
        "Risk metrics:",
    ]

    # Risk metrics
    for key in ["reid_95", "reid_50", "reid_99"]:
        val = risk_metrics.get(key)
        if val is not None:
            label = key.upper().replace("_", "")
            lines.append(f"  {label}={val:.0%}" if val <= 1 else f"  {label}={val}%")

    sr = risk_metrics.get("structural_risk")
    if sr is not None:
        lines.append(f"  Structural risk={sr:.0%}" if sr <= 1 else f"  Structural risk={sr}%")

    pattern = risk_metrics.get("risk_pattern")
    if pattern:
        lines.append(f"  Risk pattern: {pattern}")

    uniq = risk_metrics.get("uniqueness")
    if uniq is not None:
        lines.append(f"  Uniqueness rate={uniq:.0%}" if uniq <= 1 else f"  Uniqueness rate={uniq}%")

    est_supp = risk_metrics.get("estimated_suppression")
    if isinstance(est_supp, dict):
        lines.append(
            f"  Estimated suppression: "
            f"k=3: {est_supp.get(3, 0):.0%}, "
            f"k=5: {est_supp.get(5, 0):.0%}, "
            f"k=7: {est_supp.get(7, 0):.0%}")
    elif est_supp is not None:
        lines.append(f"  Estimated suppression at k=5: {est_supp:.0%}" if est_supp <= 1 else f"  Estimated suppression at k=5: {est_supp}%")

    # Combination space estimate with feasibility analysis
    combo_space = 1
    qi_cards = []
    for qi in quasi_identifiers:
        if qi in data.columns:
            card = int(data[qi].nunique())
            qi_cards.append((qi, card))
            combo_space *= card
    if qi_cards:
        target_k = risk_metrics.get("target_k", 5)
        max_combo = n_records // target_k if target_k > 0 else n_records
        lines.append(f"  Current combination space: {combo_space:,} "
                      f"(target for k={target_k}: <={max_combo:,})")
        if combo_space > max_combo:
            reduction_needed = (1 - max_combo / combo_space) * 100
            lines.append(f"  >> Need {reduction_needed:.0f}% reduction in combination space")
            # Give per-QI cardinality budget guidance
            n_qis = len(qi_cards)
            if n_qis > 0:
                # Even budget: each QI gets max_combo^(1/n_qis) unique values
                per_qi_budget = max(3, int(max_combo ** (1 / n_qis)))
                lines.append(f"  >> Even budget: ~{per_qi_budget} unique values per QI")
                # Flag which QIs are over budget
                over_budget = [(qi, c) for qi, c in qi_cards if c > per_qi_budget]
                if over_budget:
                    lines.append(
                        f"  >> Over budget: "
                        + ", ".join(f"{qi}={c}" for qi, c in over_budget))
        else:
            lines.append(f"  >> Combination space is WITHIN target — light or no preprocessing needed")

    # QI composition ratios
    n_cat_qi = n_num_qi = n_date_qi = 0
    for qi in quasi_identifiers:
        if qi not in data.columns:
            continue
        s = data[qi]
        if pd.api.types.is_datetime64_any_dtype(s):
            n_date_qi += 1
        elif pd.api.types.is_numeric_dtype(s):
            n_num_qi += 1
        else:
            # Check if string column looks like dates
            try:
                sample = s.dropna().head(20).astype(str)
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().mean() > 0.8:
                    n_date_qi += 1
                    continue
            except (ValueError, TypeError, OverflowError):
                pass  # Not a date column
            n_cat_qi += 1

    n_total_qi = max(1, len(quasi_identifiers))
    lines.append(f"  QI composition: {n_cat_qi/n_total_qi:.0%} categorical, "
                 f"{n_date_qi/n_total_qi:.0%} date, {n_num_qi/n_total_qi:.0%} numeric")

    # QI columns
    lines.append("")
    lines.append("QI columns (will be modified):")
    for i, qi in enumerate(quasi_identifiers, 1):
        if qi not in data.columns:
            continue
        series = data[qi]
        non_null = series.dropna()
        cardinality = int(series.nunique())

        # Classify type
        is_date = pd.api.types.is_datetime64_any_dtype(series)
        is_numeric = pd.api.types.is_numeric_dtype(series) and not is_date
        is_categorical = not is_numeric and not is_date
        # Check if string column is actually a date
        if is_categorical:
            try:
                sample = non_null.head(20).astype(str)
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().mean() > 0.8:
                    is_date = True
                    is_categorical = False
            except (ValueError, TypeError, OverflowError):
                pass  # Not a date column

        dtype_str = "Date/Time" if is_date else "Numeric" if is_numeric else "Categorical"

        risk_pct = risk_metrics.get("per_qi_risk", {}).get(qi, 0)
        treatment = qi_treatment.get(qi, "Standard")
        parts = [f"{i}. {qi}: {dtype_str}, {cardinality} unique, "
                 f"risk {risk_pct}%, treatment={treatment}"]

        # Distribution shape and stats for numerics
        if is_numeric and len(non_null) > 0:
            try:
                skew = float(non_null.skew())
                parts.append(f"range {non_null.min()}-{non_null.max()}")
                if abs(skew) > 2:
                    shape = "right-skewed" if skew > 0 else "left-skewed"
                elif abs(skew) < 0.5:
                    shape = "symmetric"
                else:
                    shape = "moderate-skew"
                parts.append(shape)
            except (ValueError, TypeError) as exc:
                logger.warning("[llm_method_config] Numeric stats failed for '%s': %s", qi, exc)
        elif is_categorical and len(non_null) > 0:
            # Max category frequency + rare categories
            try:
                vc = non_null.value_counts(normalize=True)
                if len(vc) > 0:
                    max_freq = float(vc.iloc[0])
                    parts.append(f"top-freq={max_freq:.0%}")
                    has_rare = int((vc < 0.01).sum())
                    if has_rare > 0:
                        parts.append(f"{has_rare} rare categories (<1%)")
            except (ValueError, TypeError) as exc:
                logger.warning("[llm_method_config] Categorical stats failed for '%s': %s", qi, exc)

        null_pct = series.isna().mean() * 100
        if null_pct > 5:
            parts.append(f"{null_pct:.0f}% null")

        lines.append(", ".join(parts))

    # Sensitive columns
    lines.append("")
    lines.append("Sensitive columns (preserved -- for context on analytical relationships):")
    for i, sc in enumerate(sensitive_columns, 1):
        if sc not in data.columns:
            continue
        series = data[sc]
        non_null = series.dropna()
        cardinality = int(series.nunique())
        risk_pct = risk_metrics.get("per_qi_risk", {}).get(sc, 0)
        dual = risk_metrics.get("dual_role_columns", {}).get(sc)

        is_cont = pd.api.types.is_numeric_dtype(series)
        sc_type = "numeric" if is_cont else "categorical"
        parts = [f"{i}. {sc}: {sc_type}, {cardinality} unique, risk {risk_pct}%"]

        if is_cont and len(non_null) > 0:
            try:
                parts.append(f"range {non_null.min()}-{non_null.max()}")
                skew = round(float(non_null.skew()), 1)
                parts.append(f"skewness={skew}")
            except (ValueError, TypeError) as exc:
                logger.warning("[llm_method_config] Sensitive stats failed for '%s': %s", sc, exc)

        if dual:
            parts.append("dual-role warning")
        lines.append(", ".join(parts))

    # Rules engine recommendation (for LLM review)
    rules_rec = risk_metrics.get("rules_recommendation")
    if rules_rec and isinstance(rules_rec, dict):
        lines.append("")
        lines.append("Rules engine recommendation (review this):")
        lines.append(f"  Method: {rules_rec.get('method', '?')}")
        if rules_rec.get('k'):
            lines.append(f"  k: {rules_rec['k']}")
        lines.append(f"  Rule: {rules_rec.get('rule', '?')}")
        lines.append(f"  Confidence: {rules_rec.get('confidence', '?')}")
        if rules_rec.get('suppression_estimate') is not None:
            lines.append(f"  Estimated suppression: {rules_rec['suppression_estimate']:.0%}")
        if rules_rec.get('reason'):
            lines.append(f"  Reason: {rules_rec['reason'][:200]}")
        lines.append(
            "  Do you agree? If not, recommend an alternative with reasoning.")

    lines.append("")
    lines.append("Recommend preprocessing steps per QI and a protection method.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main method selection function
# ---------------------------------------------------------------------------
def llm_select_method(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_columns: List[str],
    risk_metrics: Dict[str, float],
    qi_treatment: Optional[Dict[str, str]] = None,
    protection_context: str = "scientific_use",
    api_key: Optional[str] = None,
    dataset_description: Optional[str] = None,
) -> Optional[Dict]:
    """Call the Cerebras LLM for preprocessing + method recommendation.

    Args:
        data: The dataset.
        quasi_identifiers: List of QI column names.
        sensitive_columns: List of sensitive column names.
        risk_metrics: Dict with reid_95, reid_50, reid_99, structural_risk, etc.
        qi_treatment: {qi_col: 'Heavy'|'Standard'|'Light'}.
        protection_context: Protection use case.
        api_key: Optional Cerebras API key.
        dataset_description: Optional user-provided dataset description.

    Returns:
        Raw LLM response dict, or None if unavailable/failed.
    """
    try:
        assistant = get_assistant(api_key=api_key)
        if not assistant.is_available():
            logger.debug("LLM not available — skipping AI method selection")
            return None

        user_prompt = _build_dataset_profile(
            data, quasi_identifiers, sensitive_columns,
            risk_metrics, qi_treatment, protection_context,
            dataset_description=dataset_description,
        )

        result = assistant.select_method(
            user_prompt=user_prompt,
            system_prompt=_METHOD_SYSTEM_PROMPT,
        )

        if result is None:
            logger.warning("LLM method selection returned no result")
            return None

        logger.info(
            "LLM recommended method: %s (confidence: %s)",
            result.get("protection", {}).get("method", "unknown"),
            result.get("protection", {}).get("confidence", "unknown"),
        )

        return result

    except Exception as e:
        logger.warning("LLM method selection failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Response translator
# ---------------------------------------------------------------------------
def cerebras_response_to_ai_config(llm_result: Dict) -> Dict:
    """Translate the Cerebras LLM response to internal AIConfig format.

    Maps the LLM's JSON structure to the format expected by apply_ai_config().

    Args:
        llm_result: Raw dict from the LLM response.

    Returns:
        AIConfig dict with keys: preprocessing_overrides, method,
        method_confidence, method_reason, param_overrides,
        utility_guidance, attribute_disclosure, warnings, alternative.
    """
    ai_config: Dict[str, Any] = {
        "preprocessing_overrides": {},
        "method": None,
        "method_confidence": "low",
        "method_reason": "",
        "param_overrides": {},
        "utility_guidance": {},
        "attribute_disclosure": {},
        "warnings": [],
        "alternative": None,
    }

    try:
        # --- Preprocessing plan → preprocessing_overrides ---
        preprocessing = llm_result.get("preprocessing", {})
        plan = preprocessing.get("plan", [])
        for step in plan:
            qi_name = step.get("qi")
            if not qi_name:
                continue
            ai_config["preprocessing_overrides"][qi_name] = {
                "action": step.get("action", "skip"),
                "params": step.get("params", {}),
                "reasoning": step.get("reasoning", ""),
            }

        # Store expected cardinality for diagnostics
        expected_card = preprocessing.get("expected_cardinality_after", {})
        if expected_card:
            ai_config["expected_cardinality_after"] = expected_card

        # --- Protection → method + params ---
        protection = llm_result.get("protection", {})
        method = protection.get("method")
        if method:
            # Match against VALID_METHODS case-insensitively
            matched = next(
                (m for m in VALID_METHODS if m.upper() == method.upper()), None
            )
            if matched:
                ai_config["method"] = matched

        ai_config["method_confidence"] = protection.get("confidence", "low")
        ai_config["method_reason"] = protection.get("reasoning", "")
        ai_config["param_overrides"] = protection.get("params", {})

        # Estimated suppression as a diagnostic
        est_supp = protection.get("estimated_suppression")
        if est_supp:
            ai_config["estimated_suppression"] = est_supp

        # Alternative method
        alt = protection.get("alternative")
        if alt:
            alt_method = alt.get("method", "")
            alt_matched = next(
                (m for m in VALID_METHODS if m.upper() == alt_method.upper()), None
            )
            if alt_matched:
                ai_config["alternative"] = {
                    "method": alt_matched,
                    "params": alt.get("params", {}),
                    "when": alt.get("when", ""),
                }

        # --- Expected outcome → utility_guidance ---
        outcome = llm_result.get("expected_outcome", {})
        if outcome:
            ai_config["utility_guidance"] = {
                "estimated_reid_after": outcome.get("estimated_reid_after"),
                "estimated_utility": outcome.get("estimated_utility"),
                "estimated_k_achieved": outcome.get("estimated_k_achieved"),
                "key_tradeoff": outcome.get("key_tradeoff"),
            }

        # --- Attribute disclosure ---
        attr_disc = llm_result.get("attribute_disclosure", {})
        if attr_disc:
            ai_config["attribute_disclosure"] = attr_disc

        # --- Rules review (agree/disagree with rules engine) ---
        rules_review = llm_result.get("rules_review", {})
        if rules_review and isinstance(rules_review, dict):
            ai_config["rules_review"] = {
                "agrees": bool(rules_review.get("agrees", True)),
                "rules_method": rules_review.get("rules_method", ""),
                "override_reason": rules_review.get("override_reason", ""),
            }
            if not rules_review.get("agrees"):
                logger.info(
                    "LLM disagrees with rules engine (%s): %s",
                    rules_review.get("rules_method", "?"),
                    rules_review.get("override_reason", "no reason given"),
                )

        # --- Warnings ---
        warnings = llm_result.get("warnings", [])
        if isinstance(warnings, list):
            ai_config["warnings"] = warnings

    except Exception as e:
        logger.warning("Error translating LLM response to AIConfig: %s", e)
        ai_config["warnings"].append(f"Partial translation error: {e}")

    return ai_config


# ---------------------------------------------------------------------------
# Validated merge: apply_ai_config
# ---------------------------------------------------------------------------
def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep-merge override into base dict (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_ai_config(
    base_config: Dict[str, Any],
    ai_config: Optional[Dict],
    quasi_identifiers: List[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """Merge validated AI suggestions into engine config.

    The engine receives the merged config identically to user-configured
    params — it has no knowledge of AI involvement.

    Safety: AI can never reduce protection below the rules baseline.
    All invalid suggestions are ignored with warnings.

    Args:
        base_config: Config from calculate_smart_defaults() / select_method.
        ai_config: AIConfig from cerebras_response_to_ai_config(), or None.
        quasi_identifiers: List of QI column names for validation.

    Returns:
        (merged_config, warnings) tuple.
    """
    if ai_config is None:
        return base_config, []

    merged = copy.deepcopy(base_config)
    warnings: List[str] = []

    # --- Preprocessing overrides ---
    for qi, override in ai_config.get("preprocessing_overrides", {}).items():
        if qi not in quasi_identifiers:
            warnings.append(f"AI override for unknown QI '{qi}' -- ignored")
            continue

        action = override.get("action", "")
        if action not in VALID_ACTIONS:
            warnings.append(f"AI invalid action '{action}' for '{qi}' -- ignored")
            continue

        # Clamp max_categories if present
        params = override.get("params", {})
        if "max_categories" in params:
            try:
                params["max_categories"] = int(_clamp(
                    int(params["max_categories"]), "max_categories"
                ))
            except (ValueError, TypeError):
                warnings.append(f"AI invalid max_categories for '{qi}' -- ignored")
                continue

        merged.setdefault("preprocessing_overrides", {})[qi] = {
            "action": action,
            "params": params,
            "reasoning": override.get("reasoning", ""),
        }

    # --- Treatment overrides ---
    for qi, treatment in ai_config.get("treatment_overrides", {}).items():
        if qi not in quasi_identifiers:
            warnings.append(f"AI treatment for unknown QI '{qi}' -- ignored")
            continue
        if treatment not in VALID_TREATMENTS:
            warnings.append(f"AI invalid treatment '{treatment}' -- ignored")
            continue
        merged.setdefault("qi_treatment", {})[qi] = treatment

    # --- Parameter overrides (clamped to safe ranges) ---
    param_overrides = ai_config.get("param_overrides", {})
    if param_overrides:
        base_params = merged.get("method_params", {})

        # Clamp and enforce protection baseline
        for param_name in ["k", "p_change", "magnitude", "max_suppression_rate"]:
            if param_name in param_overrides:
                try:
                    ai_val = float(param_overrides[param_name])
                    clamped = _clamp(ai_val, param_name)

                    # Safety: AI cannot reduce protection below baseline
                    base_val = base_params.get(param_name)
                    if base_val is not None:
                        if param_name == "k" and clamped < float(base_val):
                            warnings.append(
                                f"AI suggested k={int(clamped)} below baseline "
                                f"k={int(base_val)} -- using baseline"
                            )
                            clamped = float(base_val)
                        elif param_name == "p_change" and clamped < float(base_val):
                            warnings.append(
                                f"AI suggested p_change={clamped:.2f} below baseline "
                                f"{float(base_val):.2f} -- using baseline"
                            )
                            clamped = float(base_val)
                        elif param_name == "magnitude" and clamped < float(base_val):
                            warnings.append(
                                f"AI suggested magnitude={clamped:.2f} below baseline "
                                f"{float(base_val):.2f} -- using baseline"
                            )
                            clamped = float(base_val)

                    if param_name == "k":
                        param_overrides[param_name] = int(clamped)
                    else:
                        param_overrides[param_name] = clamped
                except (ValueError, TypeError):
                    warnings.append(
                        f"AI invalid {param_name} value -- ignored"
                    )
                    del param_overrides[param_name]

        # Per-variable params
        per_var = param_overrides.pop("per_variable", None)
        if per_var and isinstance(per_var, dict):
            for qi, qp in per_var.items():
                if qi not in quasi_identifiers:
                    warnings.append(f"AI per-variable param for unknown QI '{qi}' -- ignored")
                    continue
                if "p_change" in qp:
                    try:
                        qp["p_change"] = _clamp(float(qp["p_change"]), "p_change")
                    except (ValueError, TypeError):
                        del qp["p_change"]
                if "noise_std" in qp:
                    try:
                        qp["noise_std"] = max(0.001, float(qp["noise_std"]))
                    except (ValueError, TypeError):
                        del qp["noise_std"]
            param_overrides["per_variable"] = per_var

        _deep_merge(merged.setdefault("method_params", {}), param_overrides)

    # --- Escalation guidance ---
    escalation = ai_config.get("escalation", {})
    if escalation:
        if "max_k" in escalation:
            try:
                escalation["max_k"] = int(_clamp(int(escalation["max_k"]), "k"))
            except (ValueError, TypeError):
                del escalation["max_k"]
        skip = escalation.get("skip_methods", [])
        escalation["skip_methods"] = [
            _match_method(m) for m in skip if _match_method(m)
        ]
        merged["escalation_limits"] = escalation

    # --- Method override (strict conditions only) ---
    ai_method = _match_method(ai_config.get("method", ""))
    ai_confidence = ai_config.get("method_confidence", "low")
    if ai_method:
        base_method = merged.get("method", "")
        if ai_method != base_method:
            if ai_confidence == "high":
                merged["ai_recommended_method"] = ai_method
                merged["ai_method_reason"] = ai_config.get("method_reason", "")
                warnings.append(
                    f"AI recommends {ai_method} instead of {base_method}: "
                    f"{ai_config.get('method_reason', '')}"
                )
            else:
                warnings.append(
                    f"AI suggests {ai_method} (confidence: {ai_confidence}) "
                    f"but rules selected {base_method} -- keeping rules choice"
                )

    # --- Utility guidance (informational only) ---
    utility = ai_config.get("utility_guidance", {})
    if utility:
        merged["ai_utility_guidance"] = utility

    # --- Attribute disclosure (informational only) ---
    attr_disc = ai_config.get("attribute_disclosure", {})
    if attr_disc:
        merged["ai_attribute_disclosure"] = attr_disc

    # --- Alternative method (informational) ---
    alt = ai_config.get("alternative")
    if alt:
        merged["ai_alternative_method"] = alt

    # --- Aggregate warnings ---
    ai_warnings = ai_config.get("warnings", [])
    if ai_warnings:
        warnings.extend([f"[AI] {w}" for w in ai_warnings])

    if warnings:
        logger.info(
            "apply_ai_config: %d AI adjustments applied, %d warnings",
            sum(1 for w in warnings if "ignored" not in w.lower()),
            len(warnings),
        )

    return merged, warnings
