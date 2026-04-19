"""Centralised column-type classification for the SDC pipeline.

Provides a single ``classify_column_type()`` function that every module
can use to get consistent semantic + structural labels like:

    'Integer — Ordinal (5 levels)'
    'Character — Address'
    'Float — Income/Financial'
    'Date — Temporal'

Previously this logic lived inside ``views/sdc_configure.py`` as private
instance methods.  Extracting it here ensures the same labels are used
everywhere (Configure table, utility metrics, relationship checks, etc.).
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from sdc_engine.sdc.config import (
    COLUMN_TYPE_KEYWORDS,
    DIRECT_IDENTIFIER_KEYWORDS,
    DIRECT_IDENTIFIER_PATTERNS,
    QI_KEYWORDS,
)


# ============================================================================
# Public API
# ============================================================================

def classify_column_type(
    var_name: str,
    col: pd.Series,
    nunique: int,
    n_rows: int,
) -> str:
    """Classify a column into a detailed semantic + structural data type.

    Combines storage type (Integer, Float, Character, Date, Boolean),
    statistical profile (Categorical, Continuous, Binary, Ordinal),
    and semantic detection (Address, Phone, Email, Name, Geographic, etc.)
    using column name keywords and value pattern matching.

    Returns a label like::

        'Integer — Ordinal (5 levels)'
        'Character — Address'
        'Character — Phone'
        'Float — Income/Financial'
        'Integer — Age (demographic)'
        'Character — Categorical — Geographic (city/region)'
    """
    var_lower = var_name.lower().replace(' ', '_').replace('-', '_')

    # 1. Semantic detection by COLUMN NAME
    semantic_tag = _detect_semantic_by_name(var_lower)

    # 2. Semantic detection by VALUE PATTERNS (for strings)
    if not semantic_tag and (
        col.dtype == 'object' or pd.api.types.is_string_dtype(col)
    ):
        semantic_tag = _detect_semantic_by_values(col)

    # 3. Structural classification (storage type + statistical profile)
    if pd.api.types.is_datetime64_any_dtype(col):
        base = 'Date — Temporal'
        if semantic_tag:
            return f'{base} ({semantic_tag})'
        return base

    elif pd.api.types.is_bool_dtype(col):
        base = 'Boolean — Binary'
        if semantic_tag:
            return f'{base} ({semantic_tag})'
        return base

    elif pd.api.types.is_integer_dtype(col):
        return _classify_integer(col, nunique, n_rows, semantic_tag)

    elif pd.api.types.is_float_dtype(col):
        return _classify_float(col, nunique, n_rows, semantic_tag)

    elif col.dtype == 'object' or pd.api.types.is_string_dtype(col):
        return _classify_string(col, nunique, n_rows, semantic_tag)

    else:
        return str(col.dtype)


def classify_columns(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> dict:
    """Classify all (or selected) columns in a DataFrame.

    Returns ``{col_name: label_str}``.
    """
    cols = columns or list(df.columns)
    n_rows = len(df)
    result = {}
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        n_unique = series.nunique()
        result[col] = classify_column_type(col, series, n_unique, n_rows)
    return result


def is_continuous_type(col_type_label: str) -> bool:
    """Check whether a column-type label indicates a continuous /
    high-cardinality numeric or temporal column.

    Useful for deciding whether eta² or Pearson r² is appropriate.
    """
    if not col_type_label:
        return False
    low = col_type_label.lower()
    if 'date' in low or 'temporal' in low:
        return True
    if 'continuous' in low:
        return True
    if any(k in low for k in ('age', 'year', 'income', 'financial')):
        return True
    return False


# ============================================================================
# Internal helpers
# ============================================================================

def _detect_semantic_by_name(var_lower: str) -> str:
    """Detect semantic type from column name using config keywords."""
    # --- Direct identifiers (name, email, phone, address, etc.) ---
    for category, keywords in DIRECT_IDENTIFIER_KEYWORDS.items():
        for kw in keywords:
            parts = var_lower.split('_')
            if kw in parts or var_lower == kw:
                label_map = {
                    'name': 'Name (direct ID)',
                    'email': 'Email (direct ID)',
                    'phone': 'Phone',
                    'address': 'Address',
                    'national_id': 'National ID',
                    'financial': 'Financial ID',
                    'medical_id': 'Medical ID',
                    'other_identifier': 'Identifier',
                }
                return label_map.get(category, category)

    # --- Definite QIs (demographic, geographic) ---
    definite_qis = QI_KEYWORDS.get('definite_qis', {})
    for kw in definite_qis:
        parts = var_lower.split('_')
        if kw in parts or var_lower == kw or var_lower.startswith(kw):
            demo_kws = {
                'age', 'gender', 'sex', 'race', 'ethnicity', 'ethnic',
                'marital', 'education', 'nationality', 'occupation',
                'job', 'profession',
            }
            geo_kws = {
                'zipcode', 'zip', 'postal', 'postcode', 'city', 'town',
                'county', 'state', 'province', 'region', 'country',
            }
            date_kws = {
                'birthdate', 'dob', 'birth_date', 'birth_year',
                'birth_month',
            }
            if kw in demo_kws:
                return f'{kw.title()} (demographic)'
            elif kw in geo_kws:
                return f'{kw.title()} (geographic)'
            elif kw in date_kws:
                return 'Birth date (temporal ID)'
            return f'{kw.title()} (quasi-identifier)'

    # --- Probable QIs (income, diagnosis, etc.) ---
    probable_qis = QI_KEYWORDS.get('probable_qis', {})
    for kw in probable_qis:
        parts = var_lower.split('_')
        if kw in parts or var_lower == kw:
            financial_kws = {'income', 'salary', 'wage'}
            medical_kws = {'diagnosis', 'icd'}
            if kw in financial_kws:
                return 'Income/Financial'
            elif kw in medical_kws:
                return 'Medical (sensitive)'
            return f'{kw.title()}'

    # --- Column type keywords (date, binary, sensitive) ---
    for category, keywords in COLUMN_TYPE_KEYWORDS.items():
        for kw in keywords:
            parts = var_lower.split('_')
            if kw in parts or var_lower == kw:
                if category == 'date':
                    return 'Date/Time'
                elif category == 'binary':
                    return 'Flag/Indicator'
                elif category == 'sensitive':
                    return f'Sensitive ({kw})'
                elif category == 'identifier':
                    return 'Identifier'
                break

    return ''


def _detect_semantic_by_values(col: pd.Series) -> str:
    """Detect semantic type by matching value patterns."""
    sample = col.dropna().astype(str)
    if len(sample) == 0:
        return ''
    sample = sample.head(200)

    best_match = ''
    best_pct = 0.0
    for pattern_name, pattern in DIRECT_IDENTIFIER_PATTERNS.items():
        try:
            matches = sample.str.match(pattern, na=False)
            pct = matches.mean()
            if pct > 0.5 and pct > best_pct:
                best_pct = pct
                label_map = {
                    'email': 'Email',
                    'phone': 'Phone',
                    'credit_card': 'Credit Card',
                    'ssn_us': 'SSN (US)',
                    'iban': 'IBAN',
                }
                best_match = label_map.get(pattern_name, pattern_name)
        except (re.error, Exception):
            continue

    if best_match:
        return best_match

    # Address detection
    addr_keywords = {
        'street', 'st', 'ave', 'avenue', 'road', 'rd', 'blvd',
        'drive', 'dr', 'lane', 'ln', 'way', 'court', 'ct',
        'place', 'pl', 'circle', 'terrace', 'highway', 'hwy',
    }
    sample_lower = sample.str.lower()
    addr_match = sample_lower.apply(
        lambda x: any(kw in str(x).split() for kw in addr_keywords)
    )
    if addr_match.mean() > 0.3:
        return 'Address'

    # Zip/postal code detection
    zip_match = sample.str.match(r'^\d{5}(-\d{4})?$', na=False)
    if zip_match.mean() > 0.5:
        return 'Zipcode (geographic)'

    # Date string detection
    date_patterns = [
        r'^\d{4}-\d{2}-\d{2}',
        r'^\d{2}/\d{2}/\d{4}',
        r'^\d{2}-\d{2}-\d{4}',
    ]
    for dp in date_patterns:
        dm = sample.str.match(dp, na=False)
        if dm.mean() > 0.5:
            return 'Date string'

    return ''


def _classify_integer(
    col: pd.Series, nunique: int, n_rows: int, semantic_tag: str,
) -> str:
    """Classify integer columns with semantic + structural detail."""
    non_null = col.dropna()
    is_ordinal = False
    if nunique <= 30 and len(non_null) > 0:
        vals = sorted(non_null.unique())
        min_v, max_v = vals[0], vals[-1]
        if min_v in (0, 1) and max_v == min_v + nunique - 1:
            is_ordinal = True

    if semantic_tag:
        if nunique <= 2:
            return f'Integer — Binary ({semantic_tag})'
        elif is_ordinal:
            return f'Integer — Ordinal ({nunique} levels, {semantic_tag})'
        elif nunique <= 20:
            return f'Integer — Categorical ({semantic_tag})'
        elif nunique > n_rows * 0.5:
            return f'Integer — Identifier ({semantic_tag})'
        else:
            return f'Integer — Continuous range ({semantic_tag})'

    if nunique <= 2:
        return 'Integer — Binary'
    elif is_ordinal and nunique <= 10:
        return f'Integer — Ordinal ({nunique} levels)'
    elif is_ordinal:
        return (f'Integer — Ordinal ({nunique} levels, '
                f'range {int(col.min())}\u2013{int(col.max())})')
    elif nunique <= 20:
        return f'Integer — Categorical ({nunique} values)'
    elif nunique > n_rows * 0.5:
        return 'Integer — Identifier-like (high cardinality)'
    else:
        return (f'Integer — Continuous '
                f'(range {int(col.min())}\u2013{int(col.max())})')


def _classify_float(
    col: pd.Series, nunique: int, n_rows: int, semantic_tag: str,
) -> str:
    """Classify float columns with semantic + structural detail."""
    if semantic_tag:
        if nunique <= 10:
            return f'Float — Categorical ({semantic_tag})'
        else:
            return f'Float — Continuous ({semantic_tag})'

    if nunique <= 10:
        return f'Float — Categorical ({nunique} values)'
    else:
        try:
            rng = f'{col.min():.1f}\u2013{col.max():.1f}'
        except Exception:
            rng = 'range'
        return f'Float — Continuous ({rng})'


def _classify_string(
    col: pd.Series, nunique: int, n_rows: int, semantic_tag: str,
) -> str:
    """Classify string/object columns with semantic + structural detail."""
    sample = col.dropna().head(200)
    sample = sample[sample.astype(str).str.strip() != '']
    numeric_pct = 0
    if len(sample) > 0:
        numeric_pct = pd.to_numeric(sample, errors='coerce').notna().mean()

    if numeric_pct <= 0.8 and len(sample) > 0:
        if sample.astype(str).str.contains(',').mean() > 0.3:
            eu_sample = (
                sample.astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
            )
            numeric_pct = (
                pd.to_numeric(eu_sample, errors='coerce').notna().mean()
            )

    if numeric_pct > 0.8:
        if semantic_tag:
            if nunique <= 20:
                return f'Char (numeric) — Coded ({semantic_tag})'
            else:
                return f'Char (numeric) — Continuous ({semantic_tag})'
        if nunique <= 20:
            return f'Char (numeric) — Coded ({nunique} values)'
        else:
            return 'Char (numeric) — Continuous'

    avg_tokens = 1.0
    if len(sample) > 0:
        avg_tokens = sample.astype(str).str.split().str.len().mean()

    if semantic_tag:
        card_hint = ''
        if nunique <= 2:
            card_hint = ', binary'
        elif nunique <= 20:
            card_hint = f', {nunique} values'
        elif nunique > n_rows * 0.5:
            card_hint = ', high cardinality'
        return f'Character — {semantic_tag}{card_hint}'

    if nunique <= 2:
        return 'Character — Binary (2 values)'
    elif nunique <= 10:
        return f'Character — Categorical ({nunique} levels)'
    elif nunique <= 20:
        return f'Character — Categorical ({nunique} values)'
    elif nunique > n_rows * 0.5:
        if avg_tokens > 3:
            return 'Character — Free text (high cardinality)'
        else:
            return 'Character — High-cardinality (ID-like)'
    elif avg_tokens > 3:
        return f'Character — Free text ({nunique} unique)'
    else:
        return f'Character — Categorical ({nunique} values, high-card)'
