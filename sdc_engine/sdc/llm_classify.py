"""
LLM-assisted column classification for SDC pipeline.

Builds column metadata, calls the Cerebras LLM for classification,
converts the response to auto_classify() format, and provides a
merge function that adds LLM signals to rule-based results.

The LLM NEVER overrides rule-based classification — it only adds
signals, boosts confidence, and surfaces warnings.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from .llm_assistant import CerebrasAssistant, get_assistant

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — generic SDC classification expertise
# Based on test_cerebras_classify.ps1, made dataset-agnostic
# ---------------------------------------------------------------------------
_CLASSIFY_SYSTEM_PROMPT = """\
You are an expert in Statistical Disclosure Control (SDC).

CRITICAL CONSTRAINT: Prefer FEWER QIs. Every column classified as QI will be MODIFIED -- its values generalized, binned, or suppressed. Too many QIs (>7) makes protection infeasible and destroys the dataset. Aim for 4-7 QIs maximum. When in doubt between QI and Sensitive, choose Sensitive with a dual-role warning and let the user decide.

Classify each column into exactly one role:

- Identifier: Directly identifies individuals, entities, or transactions. Must be removed/hashed before any analysis. Includes:
  * Person identifiers: names, tax IDs, SSNs, passport numbers, email addresses, phone numbers
  * Entity/organization identifiers: LEI (Legal Entity Identifier, ISO 17442), BIC, SWIFT codes, company registration numbers, organization codes
  * Security/instrument identifiers: ISIN (ISO 6166), CUSIP, SEDOL, ticker symbols
  * Transaction/event identifiers: high-precision timestamps (microsecond/nanosecond resolution with >30% unique values), transaction IDs, order IDs
  RULE: Columns with "lei", "cusip", "sedol" in their name are Identifiers.
  RULE: Columns with "isin" in their name are LIKELY Identifiers — but if the column is very sparse (>80% null) classify as Unassigned, and if it serves as a grouping/filtering variable (low cardinality, used to categorize rows) classify as QI instead.
  RULE: Any timestamp column with very high cardinality (>30% unique) is an Identifier (it identifies specific events, not time periods).
- QI (Quasi-Identifier): Columns that create SUBSTANTIAL re-identification risk when combined and MUST be modified. Only classify as QI if:
  (a) The column has significant risk contribution (typically >5%), AND
  (b) It could realistically be used for cross-referencing with external data sources (public records, registries, social media), AND
  (c) It is NOT the core analytical variable the dataset exists to study.
  Typical QIs: geographic location, age/date of birth, gender, trading venue, trading capacity, order side (buy/sell), instrument categories.
- Sensitive: The analytical payload -- PRESERVED for analysis. Measurements, financial values, health outcomes, quantities, percentages, scores, diagnoses, and counts. Use dual_role_warning to flag any re-identification concern instead of classifying as QI.
- Unassigned: Low analytical value AND/OR low re-identification risk. This includes:
  * Operational/behavioral metadata: order status codes, trade type codes, validity types, settlement instructions, algorithmic flags, short-selling flags
  * Near-constant columns (>90% one value)
  * Sparse columns (>50% null) unless they contain core measurements
  * Administrative codes with few categories and no external linkage potential
  * Binary flags that describe HOW something happened, not WHAT was measured

CLASSIFICATION PRINCIPLES:
1. Analytical purpose comes first. The columns the dataset exists to study MUST be Sensitive, regardless of their cardinality or risk contribution.
2. Geographic columns (municipalities, regions, districts, zip codes, states, prefectures) are ALWAYS QIs regardless of risk contribution — they are the primary linkage variables in any census or registry.
3. Temporal columns: LOW-precision dates (day/month/year) are QIs. HIGH-precision timestamps (microsecond, >30% unique) are Identifiers.
4. Demographic columns (age, gender, race, education) are common QIs.
5. High-cardinality measurements (prices, salaries, quantities with 1000+ unique values) are Sensitive. Flag with dual_role_warning if risk >5%.
6. Low risk contribution (<3%) columns should almost never be QI — EXCEPT for well-known QI types (geographic, demographic, temporal) which are QIs by nature regardless of measured risk.
7. Near-constant columns (>90% one value) should be Unassigned.
8. HIERARCHY RULE: When columns form a hierarchy (e.g., municipality within prefecture), classify ONLY the coarsest level as QI.
9. BEHAVIORAL vs SENSITIVE: Operational metadata (order_status, order_type_code, settlement_instruction, algorithmic_flag, short_selling_flag) describes PROCESS, not OUTCOME. These are Unassigned, not Sensitive. Only classify as Sensitive if the column IS the measurement the dataset exists to study.
10. SPARSE COLUMNS: Columns with >50% null values are usually conditional/supplementary fields (waiver codes, regulatory notes, reserved fields). Classify as Unassigned UNLESS the column contains continuous measurements (area m², price, quantity, percentage) — sparse measurements are still Sensitive because they represent real observations that happen to be conditional (e.g., auxiliary room area that only exists for some properties).

11. MULTILINGUAL DATASETS: Column names may be in any language. Apply the same classification logic regardless of language. Examples of common QIs in Greek: Νομός/Δήμος/Περιφέρεια (geographic), Ηλικία/Φύλο (demographic), Έτος/Ημερομηνία (temporal), Όροφος/Κατηγορία (categorical). Examples of Sensitive in Greek: Επιφάνεια (area m²), Τίμημα/Τιμή (price), Ποσοστό (percentage). Translate column names mentally to understand their semantic meaning before classifying.

RISK CONTRIBUTION: Provided from backward elimination analysis. Higher % means more contribution to re-identification risk. But high risk alone is NOT sufficient to classify as QI -- the column must also be linkable to external data AND not essential for analysis.

Respond ONLY in JSON array format. Each object must have:
- name: exact column name (preserve original name including any non-ASCII characters)
- role: identifier or qi or sensitive or unassigned
- confidence: high or medium or low
- reasoning: 1-2 sentences explaining why
- dual_role_warning: string if the column has BOTH analytical value AND re-identification risk (>5% contribution), null otherwise
- suggested_treatment: Heavy or Standard or Light or null (for QIs only -- Heavy for high-cardinality QIs needing aggressive reduction, Standard for moderate, Light for low-cardinality)

No markdown, no preamble, no explanation outside the JSON array."""


# ---------------------------------------------------------------------------
# Role normalization
# ---------------------------------------------------------------------------
_ROLE_MAP = {
    "identifier": "Identifier",
    "qi": "QI",
    "quasi-identifier": "QI",
    "quasi_identifier": "QI",
    "sensitive": "Sensitive",
    "unassigned": "Unassigned",
}

_CONFIDENCE_SCORES = {
    "high": 0.85,
    "medium": 0.60,
    "low": 0.35,
}


def _normalize_role(role: str) -> str:
    """Normalize LLM role string to auto_classify format."""
    return _ROLE_MAP.get(role.lower().strip(), "Unassigned")


def _confidence_to_score(confidence: str) -> float:
    """Convert confidence string to numeric score."""
    return _CONFIDENCE_SCORES.get(confidence.lower().strip(), 0.50)


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
# Column metadata builder
# ---------------------------------------------------------------------------
def _build_column_metadata(
    data: pd.DataFrame,
    var_priority: Dict[str, Tuple[str, float]],
) -> List[Dict[str, Any]]:
    """Build column metadata list for the LLM user prompt.

    For each column: name, dtype, cardinality, uniqueness_pct, sample_values,
    null_pct, risk_contribution_pct. Numeric cols add stats. Categorical cols
    add top values.
    """
    metadata = []
    n_rows = len(data)

    for col in data.columns:
        series = data[col]
        non_null = series.dropna()
        cardinality = int(series.nunique())

        meta: Dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "cardinality": cardinality,
            "uniqueness_pct": round(cardinality / n_rows * 100, 1) if n_rows > 0 else 0.0,
            "null_pct": round(series.isna().mean() * 100, 1),
        }

        # Risk contribution from backward elimination
        if col in var_priority:
            _priority_label, risk_pct = var_priority[col]
            meta["risk_contribution_pct"] = round(float(risk_pct), 1)
        else:
            meta["risk_contribution_pct"] = 0.0

        # Sample values (5-7 representative: common + middle + rare)
        try:
            value_counts = non_null.value_counts()
            n_vals = len(value_counts)
            sample_vals = []
            if n_vals > 0:
                sample_vals.extend(value_counts.head(3).index.tolist())
            if n_vals > 5:
                mid = n_vals // 2
                sample_vals.extend(value_counts.iloc[mid:mid + 2].index.tolist())
                sample_vals.extend(value_counts.tail(2).index.tolist())
            elif n_vals > 3:
                sample_vals.extend(value_counts.iloc[3:].index.tolist())
            # Truncate long strings, limit to 7
            meta["sample_values"] = [
                str(v)[:50] for v in sample_vals[:7]
            ]
        except Exception:
            meta["sample_values"] = []

        # Semantic type hints for domain-aware classification
        col_lower = col.lower()
        if 'lei' in col_lower:
            meta["semantic_hint"] = "Legal Entity Identifier (ISO 17442) — uniquely identifies legal entities in financial transactions"
        elif 'isin' in col_lower:
            null_pct = meta.get("null_pct", 0)
            if null_pct > 80:
                meta["semantic_hint"] = f"ISIN field but {null_pct:.0f}% null — very sparse, likely conditional/optional"
            elif cardinality < 50:
                meta["semantic_hint"] = f"ISIN field with only {cardinality} unique values — may serve as grouping variable rather than unique identifier"
            else:
                meta["semantic_hint"] = "International Securities Identification Number (ISO 6166) — uniquely identifies a security"
        elif 'cusip' in col_lower:
            meta["semantic_hint"] = "CUSIP — North American securities identifier"
        elif 'sedol' in col_lower:
            meta["semantic_hint"] = "SEDOL — London Stock Exchange securities identifier"
        elif 'bic' in col_lower or 'swift' in col_lower:
            meta["semantic_hint"] = "BIC/SWIFT code — uniquely identifies a financial institution"
        elif 'timestamp' in col_lower and meta["uniqueness_pct"] > 50:
            meta["semantic_hint"] = "High-precision timestamp (near-unique) — can identify specific events/transactions"
        elif 'timestamp' in col_lower:
            meta["semantic_hint"] = "Timestamp field — may identify events depending on precision"

        # Type-specific metadata
        if pd.api.types.is_numeric_dtype(series):
            try:
                meta["mean"] = round(float(non_null.mean()), 2)
                meta["std"] = round(float(non_null.std()), 2)
                meta["min"] = float(non_null.min())
                meta["max"] = float(non_null.max())
                meta["skewness"] = round(float(non_null.skew()), 2)
            except Exception:
                pass
        else:
            try:
                top_3 = non_null.value_counts().head(3)
                meta["top_3_values"] = {
                    str(k): int(v) for k, v in top_3.items()
                }
            except Exception:
                pass

        metadata.append(meta)

    return metadata


def _build_user_prompt(
    metadata: List[Dict[str, Any]],
    n_records: int,
    n_columns: int,
    dataset_description: Optional[str] = None,
) -> str:
    """Build the user prompt for classification from column metadata.

    Parameters
    ----------
    metadata : list of column metadata dicts
    n_records : total rows
    n_columns : total columns
    dataset_description : optional user-provided description like
        "hospital patient records" or "real estate transactions".
        When provided, the LLM can make better domain-specific decisions.
        When absent, the LLM relies purely on column metadata and
        structural signals.
    """
    if dataset_description:
        header = (
            f"Classify these {n_columns} columns from a dataset: "
            f"{dataset_description} ({n_records:,} records):"
        )
    else:
        header = (
            f"Classify these {n_columns} columns from a dataset "
            f"({n_records:,} records):"
        )

    lines = [header, ""]

    for i, meta in enumerate(metadata, 1):
        parts = [f"{i}. {meta['name']}: {meta['dtype']}, {meta['cardinality']} unique"]
        parts.append(f"risk {meta.get('risk_contribution_pct', 0.0)}%")

        uniqueness = meta.get("uniqueness_pct", 0)
        if uniqueness > 50:
            parts.append(f"uniqueness {uniqueness}%")

        if "mean" in meta:
            parts.append(f"range {meta.get('min', '?')}-{meta.get('max', '?')}")
            skew = meta.get("skewness", 0)
            if abs(skew) > 2:
                parts.append("right-skewed" if skew > 0 else "left-skewed")
        elif "top_3_values" in meta:
            top = meta["top_3_values"]
            top_str = ", ".join(f"{k}({v})" for k, v in list(top.items())[:3])
            parts.append(f"top: {top_str}")

        if meta.get("null_pct", 0) > 5:
            parts.append(f"{meta['null_pct']}% null")

        if meta.get("sample_values"):
            samples = meta["sample_values"][:5]
            parts.append(f"samples: {samples}")

        lines.append(", ".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------
def llm_classify_columns(
    data: pd.DataFrame,
    var_priority: Dict[str, Tuple[str, float]],
    api_key: Optional[str] = None,
    dataset_description: Optional[str] = None,
) -> Optional[Dict[str, Dict]]:
    """Classify columns using the Cerebras LLM.

    Args:
        data: The dataset to classify.
        var_priority: {col: (priority_label, risk_contribution_pct)} from
                      backward elimination.
        api_key: Optional Cerebras API key (falls back to env var).
        dataset_description: Optional user-provided description like
            "hospital patient records studying readmission rates".
            Helps the LLM make better domain-specific decisions.

    Returns:
        Dict in auto_classify() format: {col: {role, confidence,
        confidence_score, reason, warnings}, '_diagnostics': []},
        or None if LLM is unavailable or fails.
    """
    try:
        assistant = get_assistant(api_key=api_key)
        if not assistant.is_available():
            logger.debug("LLM not available — skipping AI classification")
            return None

        # Build metadata and prompt
        metadata = _build_column_metadata(data, var_priority)
        user_prompt = _build_user_prompt(
            metadata, len(data), len(data.columns),
            dataset_description=dataset_description,
        )

        # Call LLM
        llm_response = assistant.classify_columns(
            user_prompt=user_prompt,
            system_prompt=_CLASSIFY_SYSTEM_PROMPT,
        )
        if llm_response is None:
            logger.warning("LLM classification returned no result")
            return None

        # Convert LLM response to auto_classify format
        result: Dict[str, Dict] = {}

        for item in llm_response:
            col_name = item.get("name")
            if not col_name or col_name not in data.columns:
                logger.debug("LLM classified unknown column '%s' — skipping", col_name)
                continue

            role = _normalize_role(item.get("role", "unassigned"))
            confidence_str = item.get("confidence", "medium")
            reasoning = item.get("reasoning", "")

            warnings = []
            dual_warning = item.get("dual_role_warning")
            if dual_warning:
                warnings.append(f"[AI dual-role] {dual_warning}")

            result[col_name] = {
                "role": role,
                "confidence": confidence_str.capitalize(),
                "confidence_score": _confidence_to_score(confidence_str),
                "reason": f"[AI] {reasoning}",
                "warnings": warnings,
            }

            # Preserve suggested_treatment for potential use by method config
            treatment = item.get("suggested_treatment")
            if treatment and role == "QI":
                result[col_name]["ai_suggested_treatment"] = treatment

        result["_diagnostics"] = []

        logger.info(
            "LLM classified %d/%d columns (roles: %s)",
            len([k for k in result if k != "_diagnostics"]),
            len(data.columns),
            ", ".join(
                f"{r}={sum(1 for k, v in result.items() if k != '_diagnostics' and v.get('role') == r)}"
                for r in ["QI", "Sensitive", "Identifier", "Unassigned"]
            ),
        )

        return result

    except Exception as e:
        logger.warning("LLM classification failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Merge function
# ---------------------------------------------------------------------------
def merge_llm_with_rules(
    rules_result: Dict[str, Dict],
    llm_result: Optional[Dict[str, Dict]],
) -> Dict[str, Dict]:
    """Merge LLM classification signals into rule-based results.

    The LLM NEVER overrides rule-based classification. It can only:
    - Boost confidence when both agree
    - Add warnings when they disagree
    - Surface dual-role warnings

    Args:
        rules_result: Output from auto_classify().
        llm_result: Output from llm_classify_columns(), or None.

    Returns:
        Modified rules_result with LLM signals merged in.
    """
    if llm_result is None:
        return rules_result

    _CONFIDENCE_ORDER = {"Low": 0, "Medium": 1, "High": 2}
    _CONFIDENCE_UPGRADE = {0: "Medium", 1: "High", 2: "High"}

    for col, rules_info in rules_result.items():
        if col == "_diagnostics":
            continue
        if col not in llm_result or col == "_diagnostics":
            continue

        llm_info = llm_result[col]
        rules_role = rules_info.get("role", "Unassigned")
        llm_role = llm_info.get("role", "Unassigned")

        # Ensure warnings list exists
        if "warnings" not in rules_info:
            rules_info["warnings"] = []

        if rules_role == llm_role:
            # Agreement — boost confidence
            current_conf = rules_info.get("confidence", "Medium")
            conf_level = _CONFIDENCE_ORDER.get(current_conf, 1)
            new_conf = _CONFIDENCE_UPGRADE.get(conf_level, "High")
            rules_info["confidence"] = new_conf

            # Boost numeric score slightly
            current_score = rules_info.get("confidence_score", 0.5)
            rules_info["confidence_score"] = min(1.0, current_score + 0.10)

            # Append agreement note to reason
            rules_info["reason"] = rules_info.get("reason", "") + " | AI agrees"
        else:
            # Disagreement — add warning, do NOT override
            llm_reason = llm_info.get("reason", "").replace("[AI] ", "")
            warning = (
                f"AI suggests {llm_role} instead of {rules_role}: {llm_reason}"
            )
            rules_info["warnings"].append(warning)

            # Special note for sensitive vs QI disagreement
            if (rules_role == "Sensitive" and llm_role == "QI") or \
               (rules_role == "QI" and llm_role == "Sensitive"):
                rules_info["warnings"].append(
                    "Consider reviewing this column's role — "
                    "rules and AI disagree on QI vs Sensitive classification"
                )

        # Always merge dual-role warnings from LLM
        for w in llm_info.get("warnings", []):
            if w not in rules_info["warnings"]:
                rules_info["warnings"].append(w)

        # Carry over AI treatment suggestion (Heavy/Standard/Light)
        ai_treatment = llm_info.get("ai_suggested_treatment")
        if ai_treatment:
            rules_info["ai_suggested_treatment"] = ai_treatment

    # Merge diagnostics
    llm_diag = llm_result.get("_diagnostics", [])
    if llm_diag and "_diagnostics" in rules_result:
        rules_result["_diagnostics"].extend(llm_diag)

    return rules_result


def merge_rules_into_llm(
    llm_result: Dict[str, Dict],
    rules_result: Optional[Dict[str, Dict]],
) -> Dict[str, Dict]:
    """Merge rule-based signals into LLM-primary results.

    .. note::
        Currently unused — the UI always uses ``merge_llm_with_rules()``
        (rules primary).  Kept for a future "AI-first" classification mode.
        # TODO: wire up for AI-first mode or remove if not needed.

    LLM classification is kept as the primary role assignment.
    Rules engine adds:
    - Confidence boost when both agree
    - Warnings when they disagree (rules suggest different role)
    - Cross-column diagnostics (LLM doesn't produce these)
    - Coverage for columns the LLM may have missed

    Args:
        llm_result: Output from llm_classify_columns() (primary).
        rules_result: Output from auto_classify() (secondary), or None.

    Returns:
        Modified llm_result with rules signals merged in.
    """
    if rules_result is None:
        return llm_result

    _CONFIDENCE_ORDER = {"Low": 0, "Medium": 1, "High": 2}
    _CONFIDENCE_UPGRADE = {0: "Medium", 1: "High", 2: "High"}

    # Add columns that LLM missed but rules classified
    for col, rules_info in rules_result.items():
        if col == "_diagnostics":
            continue
        if col not in llm_result:
            # LLM missed this column — use rules classification with a note
            entry = dict(rules_info)
            entry["reason"] = entry.get("reason", "") + " [rules fallback -- AI did not classify]"
            llm_result[col] = entry
            continue

    # Enrich LLM results with rules signals
    for col, llm_info in llm_result.items():
        if col == "_diagnostics":
            continue
        if col not in rules_result or col == "_diagnostics":
            continue

        rules_info = rules_result[col]
        llm_role = llm_info.get("role", "Unassigned")
        rules_role = rules_info.get("role", "Unassigned")

        if "warnings" not in llm_info:
            llm_info["warnings"] = []

        if llm_role == rules_role:
            # Agreement — boost confidence
            current_conf = llm_info.get("confidence", "Medium")
            conf_level = _CONFIDENCE_ORDER.get(current_conf, 1)
            new_conf = _CONFIDENCE_UPGRADE.get(conf_level, "High")
            llm_info["confidence"] = new_conf
            llm_info["confidence_score"] = min(
                1.0, llm_info.get("confidence_score", 0.5) + 0.10
            )
            llm_info["reason"] = llm_info.get("reason", "") + " | Rules agree"
        else:
            # Disagreement — add rules warning, keep LLM role
            rules_reason = rules_info.get("reason", "")
            llm_info["warnings"].append(
                f"Rules engine suggests {rules_role} instead of {llm_role}: "
                f"{rules_reason}"
            )
            if (llm_role == "Sensitive" and rules_role == "QI") or \
               (llm_role == "QI" and rules_role == "Sensitive"):
                llm_info["warnings"].append(
                    "Consider reviewing -- AI and rules disagree on "
                    "QI vs Sensitive classification"
                )

        # Merge any rules warnings not already present
        for w in rules_info.get("warnings", []):
            if w not in llm_info["warnings"]:
                llm_info["warnings"].append(w)

    # Carry over cross-column diagnostics from rules (LLM doesn't produce these)
    rules_diag = rules_result.get("_diagnostics", [])
    if rules_diag:
        llm_result.setdefault("_diagnostics", []).extend(rules_diag)

    return llm_result
