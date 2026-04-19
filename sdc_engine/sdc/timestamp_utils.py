"""
Timestamp Truncation Utilities
==============================

Grain-aware timestamp generalization for date/time QI columns.
Supports ISO 8601 strings at multiple granularities:
minute, 5min, hour, date, week, month, quarter, year.

Ported from v2-generalized-pipeline (script2_scenarios.py + script4_execute.py).
"""

from typing import Dict, List, Optional, Tuple


# ── Adaptive grain tables ─────────────────────────────────────────────
# Each table lists (grain_code, description) tuples from finest to coarsest.

GRAIN_TABLES: Dict[str, List[Tuple[str, str]]] = {
    "intraday": [
        ("raw", "No generalization (keep raw timestamp)"),
        ("minute", "Truncate to minute"),
        ("5min", "Truncate to 5-minute bin"),
        ("hour", "Truncate to hour"),
    ],
    "daily": [
        ("raw", "No generalization (keep raw timestamp)"),
        ("hour", "Truncate to hour"),
        ("date", "Truncate to date"),
    ],
    "monthly": [
        ("raw", "No generalization (keep raw timestamp)"),
        ("date", "Truncate to date"),
        ("week", "Truncate to week"),
        ("month", "Truncate to month"),
    ],
    "yearly": [
        ("raw", "No generalization (keep raw timestamp)"),
        ("month", "Truncate to month"),
        ("quarter", "Truncate to quarter"),
        ("year", "Truncate to year"),
    ],
}


def resolve_grain(
    ts_span_days: Optional[float],
    grain_config: str = "auto",
) -> Tuple[str, List[Tuple[str, str]], Optional[str]]:
    """Select the appropriate grain table based on time span.

    Parameters
    ----------
    ts_span_days : float or None
        Span of the timestamp column in days.
    grain_config : str
        ``"auto"`` to detect from span, or a key of ``GRAIN_TABLES``.

    Returns
    -------
    (grain_name, level_tuples, note_or_None)
    """
    if grain_config != "auto":
        if grain_config in GRAIN_TABLES:
            return grain_config, GRAIN_TABLES[grain_config], None
        grain_config = "auto"  # fallback

    if ts_span_days is None or ts_span_days < 0:
        return "yearly", GRAIN_TABLES["yearly"], \
            "ts_span_days unavailable — defaulting to yearly grain"
    if ts_span_days <= 1:
        return "intraday", GRAIN_TABLES["intraday"], None
    if ts_span_days <= 31:
        return "daily", GRAIN_TABLES["daily"], None
    if ts_span_days <= 365:
        return "monthly", GRAIN_TABLES["monthly"], None
    return "yearly", GRAIN_TABLES["yearly"], None


def truncate_timestamp(value: str, grain: str) -> str:
    """Truncate an ISO 8601 timestamp string to the given grain.

    Parameters
    ----------
    value : str
        Timestamp string, e.g. ``"2024-03-15T14:23:45Z"``.
    grain : str
        One of: ``raw``, ``suppress``, ``minute``, ``5min``, ``hour``,
        ``date``, ``week``, ``month``, ``quarter``, ``year``.

    Returns
    -------
    str
        Truncated timestamp string.
    """
    if not value or grain == "raw":
        return value

    # Bare-time strings (e.g. "T14:23:45")
    if "T" in value and (value.startswith("T") or not value.split("T")[0]):
        return ""

    # Sentinel dates (year >= 9000 treated as missing)
    try:
        year = int(value[:4])
        if year >= 9000:
            return value
    except (ValueError, IndexError):
        return ""

    if grain == "suppress":
        return ""
    if grain == "minute":
        return value[:16] + "Z" if len(value) >= 16 else value
    if grain == "5min":
        try:
            minute = (int(value[14:16]) // 5) * 5
            return value[:14] + f"{minute:02d}Z"
        except (ValueError, IndexError):
            return value
    if grain == "hour":
        return value[:13] + "Z" if len(value) >= 13 else value
    if grain == "date":
        return value[:10]
    if grain == "week":
        # Truncate to ISO week start (Monday)
        try:
            from datetime import datetime as dt
            d = dt.fromisoformat(value[:10])
            monday = d - __import__('datetime').timedelta(days=d.weekday())
            return monday.strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            return value[:10]
    if grain == "month":
        return value[:7]
    if grain == "quarter":
        try:
            month = int(value[5:7])
            q = ((month - 1) // 3) * 3 + 1
            return f"{value[:5]}{q:02d}"
        except (ValueError, IndexError):
            return value
    if grain == "year":
        return value[:4]
    return value


def level_to_grain(level_desc: str) -> str:
    """Map a human-readable level description to a grain code.

    Parameters
    ----------
    level_desc : str
        Description like ``"Truncate to hour"`` or ``"No generalization"``.

    Returns
    -------
    str
        Grain code (``raw``, ``minute``, ``5min``, ``hour``, etc.).
    """
    desc = level_desc.lower()
    if "no generalization" in desc or "keep raw" in desc:
        return "raw"
    if "suppress" in desc or "drop" in desc:
        return "suppress"
    if "5-minute" in desc or "5-min" in desc:
        return "5min"
    if "minute" in desc:
        return "minute"
    if "hour" in desc:
        return "hour"
    if "date" in desc or "day" in desc:
        return "date"
    if "week" in desc:
        return "week"
    if "month" in desc:
        return "month"
    if "quarter" in desc:
        return "quarter"
    if "year" in desc:
        return "year"
    return "raw"
