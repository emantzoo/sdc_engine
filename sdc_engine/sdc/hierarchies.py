"""
Smart Hierarchy Builders (ARX-inspired)
========================================

Multi-level generalization hierarchies for quasi-identifier columns.
Inspired by ARX Data Anonymization Tool's hierarchy builder architecture.

Each hierarchy defines multiple generalization levels (0 = original, N = most
general).  Generalization at any level is a single vectorized ``series.map()``
call — O(n) array lookup, no per-value computation.

Four builder classes handle different data types:

* **IntervalHierarchyBuilder** — numeric data (ages, income, measurements)
* **DateHierarchyBuilder** — temporal data (dates, timestamps)
* **MaskingHierarchyBuilder** — alphanumeric codes (postal codes, phone numbers)
* **CategoricalHierarchyBuilder** — nominal categories (cities, occupations)

The ``build_hierarchy_for_column()`` factory auto-selects the right builder
based on ``_classify_qi_type()`` output from ``smart_defaults.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ============================================================================
# Core Hierarchy class
# ============================================================================

class Hierarchy:
    """Multi-level generalization hierarchy for a single column.

    Parameters
    ----------
    column_name : str
        Name of the column this hierarchy applies to.
    levels : list of dict
        ``levels[0]`` is the identity mapping (every value maps to itself).
        ``levels[i]`` maps ``original_value -> generalized_value`` at level *i*.
    builder_type : str
        Name of the builder that created this hierarchy (for diagnostics).
    """

    def __init__(
        self,
        column_name: str,
        levels: List[Dict[str, str]],
        builder_type: str = 'manual',
    ):
        self.column_name = column_name
        self.levels = levels
        self.builder_type = builder_type
        # Pre-compute cardinalities and info loss per level
        self._cardinalities: List[int] = []
        self._info_loss: List[float] = []
        for lvl_map in levels:
            card = len(set(lvl_map.values())) if lvl_map else 0
            self._cardinalities.append(card)
        card_0 = self._cardinalities[0] if self._cardinalities else 1
        for card in self._cardinalities:
            # ARX "Loss" metric: 1 - (cardinality_at_level / cardinality_at_0)
            self._info_loss.append(
                1.0 - (card / card_0) if card_0 > 0 else 0.0
            )

    @property
    def max_level(self) -> int:
        """Highest generalization level (0-indexed)."""
        return max(0, len(self.levels) - 1)

    def cardinality_at(self, level: int) -> int:
        """Number of distinct generalized values at *level*."""
        level = min(level, self.max_level)
        return self._cardinalities[level]

    def info_loss_at(self, level: int) -> float:
        """Information loss at *level*: ``1 - card(level)/card(0)``."""
        level = min(level, self.max_level)
        return self._info_loss[level]

    def generalize(self, series: pd.Series, level: int) -> pd.Series:
        """Apply generalization at *level* to *series*.  Vectorized map."""
        level = min(level, self.max_level)
        if level <= 0:
            return series.copy()
        mapping = self.levels[level]
        # Mapping keys are always strings.  Convert to string first so
        # the map lookup succeeds for numeric and datetime dtypes.
        if (pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series)):
            str_series = series.astype(str)
            result = str_series.map(mapping)
            # Restore NaN where original was NaN
            result[series.isna()] = None
        else:
            result = series.map(mapping)
        # Values not in mapping (e.g. NaN, unseen) fall through unchanged
        mask = result.isna() & series.notna()
        if mask.any():
            result[mask] = series[mask].astype(str)
        return result

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for caching / JSON storage."""
        return {
            'column_name': self.column_name,
            'levels': self.levels,
            'builder_type': self.builder_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Hierarchy':
        """Deserialise from dict."""
        return cls(
            column_name=d['column_name'],
            levels=d['levels'],
            builder_type=d.get('builder_type', 'manual'),
        )

    @classmethod
    def from_legacy_dict(cls, column_name: str, mapping: Dict[str, str]) -> 'Hierarchy':
        """Wrap the existing ``{val: gen_val}`` format as a 2-level hierarchy.

        Level 0 = identity, level 1 = the provided mapping.  Backward-compat
        bridge so old ``gen_config['hierarchies']`` dicts still work.
        """
        all_vals = set(mapping.keys())
        identity = {v: v for v in all_vals}
        return cls(
            column_name=column_name,
            levels=[identity, mapping],
            builder_type='legacy_dict',
        )

    def __repr__(self) -> str:
        return (
            f"Hierarchy('{self.column_name}', builder='{self.builder_type}', "
            f"levels={len(self.levels)}, card=[{', '.join(str(c) for c in self._cardinalities)}])"
        )


# ============================================================================
# Helper: nice interval boundaries
# ============================================================================

def _nice_number(x: float, round_up: bool = True) -> int:
    """Round *x* to a 'nice' number (1, 2, 5, 10, 20, 25, 50, 100, ...)."""
    if x <= 0:
        return 1
    exp = math.floor(math.log10(x))
    frac = x / (10 ** exp)
    nice_fracs = [1, 2, 2.5, 5, 10]
    if round_up:
        for n in nice_fracs:
            if n >= frac:
                return max(1, int(n * (10 ** exp)))
        return max(1, int(10 * (10 ** exp)))
    else:
        prev = 1
        for n in nice_fracs:
            if n > frac:
                return max(1, int(prev * (10 ** exp)))
            prev = n
        return max(1, int(10 * (10 ** exp)))


# ============================================================================
# IntervalHierarchyBuilder (numeric data)
# ============================================================================

class IntervalHierarchyBuilder:
    """Build multi-level interval hierarchies for numeric columns.

    Parameters
    ----------
    n_levels : int
        Number of generalization levels (excluding identity level 0).
    base_interval : int or None
        Width of intervals at level 1.  Auto-computed if None.
    top_code : float or None
        Values above this are replaced with ``">=top_code"``.
    bottom_code : float or None
        Values below this are replaced with ``"<=bottom_code"``.
    quantile_mode : bool or None
        If True, use quantile boundaries.  If None (default), auto-detect
        from data skewness (|skew| > 3 triggers quantile mode).
    """

    def __init__(
        self,
        n_levels: int = 4,
        base_interval: Optional[int] = None,
        top_code: Optional[float] = None,
        bottom_code: Optional[float] = None,
        quantile_mode: Optional[bool] = None,
    ):
        self.n_levels = max(1, n_levels)
        self.base_interval = base_interval
        self.top_code = top_code
        self.bottom_code = bottom_code
        self.quantile_mode = quantile_mode

    def build(self, series: pd.Series, column_name: str) -> Hierarchy:
        """Build the hierarchy from observed data values."""
        valid = series.dropna()
        if len(valid) == 0:
            return Hierarchy(column_name, [{}], builder_type='interval')

        # Determine if quantile mode should be used
        use_quantile = self.quantile_mode
        if use_quantile is None:
            try:
                skew = float(valid.skew())
                use_quantile = abs(skew) > 3.0
            except (ValueError, TypeError) as exc:
                log.warning("[Hierarchy] Skewness calculation failed for '%s': %s", column_name, exc)
                use_quantile = False

        # Collect all unique original values (as strings for mapping keys)
        try:
            numeric_vals = pd.to_numeric(valid, errors='coerce').dropna()
        except (ValueError, TypeError) as exc:
            log.warning("[Hierarchy] Numeric coercion failed for '%s': %s", column_name, exc)
            numeric_vals = valid

        if len(numeric_vals) == 0:
            return Hierarchy(column_name, [{}], builder_type='interval')

        # Apply top/bottom coding
        vals_arr = numeric_vals.values.copy().astype(float)
        tc = self.top_code
        bc = self.bottom_code
        if tc is None:
            tc = float(np.percentile(vals_arr, 98))
        if bc is None:
            bc = float(np.percentile(vals_arr, 2))

        vmin, vmax = float(np.min(vals_arr)), float(np.max(vals_arr))

        # Determine base interval
        base = self.base_interval
        if base is None:
            data_range = max(vmax - vmin, 1)
            base = _nice_number(data_range / 50, round_up=True)
            # For age-like data (0-120 range), prefer 5-year bins
            if 0 <= vmin and vmax <= 135 and data_range > 10:
                base = 5

        # Build levels
        levels = []
        all_original_strs = sorted(set(str(v) for v in numeric_vals.unique()))

        # Level 0: identity
        identity = {s: s for s in all_original_strs}
        levels.append(identity)

        for lvl in range(1, self.n_levels + 1):
            interval_width = base * (2 ** (lvl - 1))
            mapping = {}

            if use_quantile and lvl == 1:
                # Quantile-based boundaries for first level
                n_bins = max(2, int(len(set(vals_arr)) / (2 ** (lvl - 1))))
                n_bins = min(n_bins, 50)
                try:
                    _, bin_edges = pd.qcut(vals_arr, q=n_bins, retbins=True, duplicates='drop')
                except (ValueError, IndexError) as exc:
                    log.warning("[Hierarchy] Quantile binning failed for '%s': %s — falling back to linspace", column_name, exc)
                    bin_edges = np.linspace(vmin, vmax, n_bins + 1)
                for v_str in all_original_strs:
                    v = float(v_str) if v_str.replace('.', '', 1).replace('-', '', 1).isdigit() else 0
                    idx = np.searchsorted(bin_edges[1:], v, side='right')
                    idx = min(idx, len(bin_edges) - 2)
                    lo = bin_edges[idx]
                    hi = bin_edges[min(idx + 1, len(bin_edges) - 1)]
                    mapping[v_str] = f"{int(lo)}-{int(hi)}" if lo == int(lo) and hi == int(hi) else f"{lo:.1f}-{hi:.1f}"
            else:
                # Equal-width interval boundaries
                for v_str in all_original_strs:
                    try:
                        v = float(v_str)
                    except (ValueError, TypeError):
                        mapping[v_str] = '*'
                        continue

                    if v > tc:
                        mapping[v_str] = f">={int(tc)}" if tc == int(tc) else f">={tc:.1f}"
                    elif v < bc:
                        mapping[v_str] = f"<={int(bc)}" if bc == int(bc) else f"<={bc:.1f}"
                    else:
                        bin_lo = int(v // interval_width) * interval_width
                        bin_hi = bin_lo + interval_width - 1
                        # Use int formatting when values are integer-like
                        if interval_width == int(interval_width):
                            mapping[v_str] = f"{int(bin_lo)}-{int(bin_hi)}"
                        else:
                            mapping[v_str] = f"{bin_lo:.1f}-{bin_hi:.1f}"

            levels.append(mapping)

        # Final level: everything maps to "*"
        if len(levels) <= self.n_levels:
            levels.append({s: '*' for s in all_original_strs})

        return Hierarchy(column_name, levels, builder_type='interval')


# ============================================================================
# DateHierarchyBuilder (temporal data)
# ============================================================================

class DateHierarchyBuilder:
    """Build multi-level date hierarchies.

    Granularity levels: exact → day → week → month → quarter → year → ``*``

    Parameters
    ----------
    granularities : list of str or None
        Ordered granularity names from fine to coarse.
        Default: ``['day', 'month', 'quarter', 'year']``.
    """

    _GRAN_ORDER = ['day', 'week', 'month', 'quarter', 'year']

    def __init__(self, granularities: Optional[List[str]] = None):
        self.granularities = granularities or ['day', 'month', 'quarter', 'year']

    def build(self, series: pd.Series, column_name: str) -> Hierarchy:
        """Build the hierarchy from observed datetime values."""
        valid = series.dropna()
        if len(valid) == 0:
            return Hierarchy(column_name, [{}], builder_type='date')

        # Try to parse as datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(valid):
            try:
                valid = pd.to_datetime(valid, errors='coerce', dayfirst=True).dropna()
            except (ValueError, TypeError, OverflowError) as exc:
                log.warning("[Hierarchy] Date parsing (dayfirst) failed for '%s': %s", column_name, exc)
                try:
                    valid = pd.to_datetime(valid, errors='coerce').dropna()
                except (ValueError, TypeError, OverflowError) as exc2:
                    log.warning("[Hierarchy] Date parsing fallback failed for '%s': %s", column_name, exc2)
                    return Hierarchy(column_name, [{}], builder_type='date')

        if len(valid) == 0:
            return Hierarchy(column_name, [{}], builder_type='date')

        # Collect original string representations using the same format
        # as Series.astype(str) — which is what generalize() uses for lookup.
        # For datetime, pd.Series.astype(str) omits '00:00:00' time component.
        orig_series = series.dropna()
        orig_strs = sorted(set(orig_series.astype(str).unique()))

        # Build a datetime lookup for original strings
        dt_lookup = {}
        for s in orig_strs:
            try:
                dt_lookup[s] = pd.Timestamp(s)
            except (ValueError, TypeError, OverflowError):
                try:
                    dt_lookup[s] = pd.to_datetime(s, dayfirst=True)
                except (ValueError, TypeError, OverflowError):
                    dt_lookup[s] = None

        # Level 0: identity
        identity = {s: s for s in orig_strs}
        levels = [identity]

        # Auto-detect sensible granularities based on date span
        date_range_days = (valid.max() - valid.min()).days if len(valid) > 1 else 0
        active_grans = []
        for g in self.granularities:
            if g == 'day' and date_range_days < 90:
                continue  # Skip day level if data spans < 3 months
            if g == 'week' and date_range_days < 180:
                continue
            active_grans.append(g)

        if not active_grans:
            active_grans = ['year']

        for gran in active_grans:
            mapping = {}
            for s, dt in dt_lookup.items():
                if dt is None:
                    mapping[s] = s
                    continue
                if gran == 'day':
                    mapping[s] = dt.strftime('%Y-%m-%d')
                elif gran == 'week':
                    mapping[s] = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
                elif gran == 'month':
                    mapping[s] = dt.strftime('%Y-%m')
                elif gran == 'quarter':
                    q = (dt.month - 1) // 3 + 1
                    mapping[s] = f"{dt.year}-Q{q}"
                elif gran == 'year':
                    mapping[s] = str(dt.year)
            levels.append(mapping)

        # Final: suppress all
        levels.append({s: '*' for s in orig_strs})

        return Hierarchy(column_name, levels, builder_type='date')


# ============================================================================
# MaskingHierarchyBuilder (alphanumeric codes)
# ============================================================================

class MaskingHierarchyBuilder:
    """Build masking hierarchies for alphanumeric codes (postal, phone, ID).

    Progressive character masking from one direction.

    Parameters
    ----------
    mask_char : str
        Character used for masking (default ``'*'``).
    direction : str
        ``'right'`` (default, natural for postal codes) or ``'left'``.
    """

    def __init__(self, mask_char: str = '*', direction: str = 'right'):
        self.mask_char = mask_char
        self.direction = direction

    def build(self, series: pd.Series, column_name: str) -> Hierarchy:
        """Build the hierarchy from observed code values."""
        valid = series.dropna().astype(str)
        if len(valid) == 0:
            return Hierarchy(column_name, [{}], builder_type='masking')

        orig_strs = sorted(set(valid.unique()))
        max_len = max(len(s) for s in orig_strs) if orig_strs else 0

        if max_len == 0:
            return Hierarchy(column_name, [{}], builder_type='masking')

        # Level 0: identity
        identity = {s: s for s in orig_strs}
        levels = [identity]

        # One level per character masked
        for n_masked in range(1, max_len + 1):
            mapping = {}
            for s in orig_strs:
                slen = len(s)
                if n_masked >= slen:
                    mapping[s] = self.mask_char * slen
                elif self.direction == 'right':
                    mapping[s] = s[:slen - n_masked] + self.mask_char * n_masked
                else:  # left
                    mapping[s] = self.mask_char * n_masked + s[n_masked:]
            levels.append(mapping)

        return Hierarchy(column_name, levels, builder_type='masking')


# ============================================================================
# CategoricalHierarchyBuilder (nominal categories)
# ============================================================================

class CategoricalHierarchyBuilder:
    """Build conservative categorical hierarchies.

    Uses a simple, predictable strategy:

    * **Level 1** — merge categories below ``min_frequency`` into ``"Other"``
    * **Level 2** — keep top-K categories, rest become ``"Other"``
    * **Level 3** — suppress all (``"*"``)

    User-provided semantic groupings (e.g. ``{"Northern": ["N", "NE", "NW"]}``)
    take precedence over auto-grouping.

    Parameters
    ----------
    groupings : dict or None
        Optional user-defined semantic groups:
        ``{group_label: [value1, value2, ...]}``.
    min_frequency : float
        Categories appearing in fewer than this fraction of records are
        merged into ``"Other"`` at level 1.  Default 0.01 (1%).
    top_k : int or None
        Number of categories to keep at level 2.  Auto-computed if None.
    """

    def __init__(
        self,
        groupings: Optional[Dict[str, List[str]]] = None,
        min_frequency: float = 0.01,
        top_k: Optional[int] = None,
    ):
        self.groupings = groupings
        self.min_frequency = min_frequency
        self.top_k = top_k

    def build(self, series: pd.Series, column_name: str) -> Hierarchy:
        """Build the hierarchy from observed categorical values."""
        valid = series.dropna().astype(str)
        if len(valid) == 0:
            return Hierarchy(column_name, [{}], builder_type='categorical')

        orig_strs = sorted(set(valid.unique()))
        n_total = len(valid)

        # Level 0: identity
        identity = {s: s for s in orig_strs}
        levels = [identity]

        if self.groupings:
            # User-provided semantic groupings
            # Level 1: apply groupings
            val_to_group = {}
            for group_label, members in self.groupings.items():
                for m in members:
                    val_to_group[str(m)] = group_label

            mapping_1 = {}
            for s in orig_strs:
                mapping_1[s] = val_to_group.get(s, s)  # unmapped stay as-is
            levels.append(mapping_1)

            # Level 2: merge remaining ungrouped into "Other"
            grouped_vals = set(val_to_group.keys())
            mapping_2 = {}
            for s in orig_strs:
                if s in grouped_vals:
                    mapping_2[s] = val_to_group[s]
                else:
                    mapping_2[s] = 'Other'
            levels.append(mapping_2)
        else:
            # Auto-grouping: conservative frequency-based approach
            value_counts = valid.value_counts()
            min_count = max(1, int(n_total * self.min_frequency))

            # Level 1: merge rare categories into "Other"
            mapping_1 = {}
            for s in orig_strs:
                count = value_counts.get(s, 0)
                mapping_1[s] = s if count >= min_count else 'Other'
            levels.append(mapping_1)

            # Level 2: keep top-K categories
            top_k = self.top_k
            if top_k is None:
                # Auto: keep ~sqrt(n_unique) but at least 3, at most 10
                n_unique = len(orig_strs)
                top_k = max(3, min(10, int(math.sqrt(n_unique))))

            top_categories = value_counts.nlargest(top_k).index.tolist()
            top_set = set(str(c) for c in top_categories)
            mapping_2 = {}
            for s in orig_strs:
                mapping_2[s] = s if s in top_set else 'Other'
            levels.append(mapping_2)

        # Final level: suppress all
        levels.append({s: '*' for s in orig_strs})

        return Hierarchy(column_name, levels, builder_type='categorical')


# ============================================================================
# Factory: auto-select and build hierarchy for a column
# ============================================================================

def build_hierarchy_for_column(
    col: str,
    data: pd.DataFrame,
    column_types: Optional[Dict[str, str]] = None,
    user_hierarchy: Optional[Union[Dict[str, str], Hierarchy]] = None,
) -> Optional[Hierarchy]:
    """Auto-select the right builder and create a hierarchy for *col*.

    Parameters
    ----------
    col : str
        Column name.
    data : pd.DataFrame
        Dataset (must contain *col*).
    column_types : dict or None
        Semantic type labels from the Configure table (e.g.
        ``{'age': 'Integer - Age (demographic)'}``).
    user_hierarchy : dict or Hierarchy or None
        User-provided hierarchy.  If a ``Hierarchy`` object, returned as-is.
        If a plain dict ``{val: gen_val}``, wrapped via ``from_legacy_dict``.

    Returns
    -------
    Hierarchy or None
        ``None`` for binary columns (no hierarchy useful).
    """
    # User-provided takes precedence
    if isinstance(user_hierarchy, Hierarchy):
        return user_hierarchy
    if isinstance(user_hierarchy, dict) and user_hierarchy:
        return Hierarchy.from_legacy_dict(col, user_hierarchy)

    if col not in data.columns:
        return None

    series = data[col]
    n_unique = series.nunique()

    # Skip binary columns — no useful hierarchy
    if n_unique <= 2:
        return None

    # Classify column type
    from sdc_engine.sdc.smart_defaults import _classify_qi_type
    type_label = (column_types or {}).get(col, '')
    ct = _classify_qi_type(col, data, type_label)

    # Skip id-like, free text, binary
    if ct['is_id_like'] or ct['is_free_text'] or ct['is_binary']:
        return None

    # Route to appropriate builder
    if ct['is_date']:
        builder = DateHierarchyBuilder()
        return builder.build(series, col)

    if ct['is_geo']:
        # Check if it looks like a postal/numeric code
        sample = series.dropna().astype(str).head(100)
        digit_ratio = sample.str.match(r'^\d+$').mean()
        if digit_ratio > 0.8:
            # Postal code-like: use masking
            builder = MaskingHierarchyBuilder()
            return builder.build(series, col)
        else:
            # Geographic names: use categorical with conservative grouping
            builder = CategoricalHierarchyBuilder()
            return builder.build(series, col)

    if ct['is_numeric'] or ct['is_age']:
        builder = IntervalHierarchyBuilder()
        return builder.build(series, col)

    # Default: categorical
    if n_unique > 2:
        builder = CategoricalHierarchyBuilder()
        return builder.build(series, col)

    return None
