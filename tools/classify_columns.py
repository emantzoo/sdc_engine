#!/usr/bin/env python3
"""
Standalone Column Type & Direct Identifier Classifier
======================================================

Fully self-contained — no imports from the SDC engine.
All detection logic is duplicated from the SDC pipeline modules:

  - classify_column_type()       from  sdc/column_types.py
  - identify_column_types()      from  sdc/detection/column_types.py
  - auto_detect_direct_identifiers()  from  sdc/detection/column_types.py
  - detect_greek_identifiers()   from  sdc/sdc_preprocessing.py
  - keyword dicts                from  sdc/config.py

Usage:
    python classify_columns.py data.csv
    python classify_columns.py data.xlsx --sheet "Sheet2"
    python classify_columns.py data.parquet --sample 5000
    python classify_columns.py data.csv --json
    python classify_columns.py data.csv --out report.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ============================================================================
# Config  (duplicated from sdc_engine/sdc/config.py)
# ============================================================================

DIRECT_IDENTIFIER_KEYWORDS: Dict[str, List[str]] = {
    'name': [
        'name', 'firstname', 'first_name', 'lastname', 'last_name',
        'surname', 'fullname', 'full_name', 'middlename', 'middle_name',
    ],
    'email': ['email', 'mail', 'e_mail', 'email_address'],
    'phone': ['phone', 'telephone', 'mobile', 'cell', 'fax', 'tel'],
    'address': [
        'address', 'street', 'addr', 'residence',
        'home_address', 'work_address', 'postal_address',
    ],
    'national_id': [
        'ssn', 'social_security', 'nin', 'national_id', 'national_insurance',
        'passport', 'passport_no', 'driver_license', 'drivers_license',
        'license_no', 'tax_id', 'tin', 'vat', 'afm', 'amka',
    ],
    'financial': [
        'credit_card', 'card_number', 'account_number', 'iban',
        'bank_account', 'cvv', 'pin',
    ],
    'medical_id': [
        'medical_record', 'health_id', 'patient_id', 'mrn',
        'medical_record_number', 'health_record',
    ],
    'other_identifier': [
        'id', 'uuid', 'guid', 'record_id', 'row_id', 'key',
        'index_id', 'unique_id', 'identifier',
    ],
}

DIRECT_IDENTIFIER_PATTERNS: Dict[str, str] = {
    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    # Phone: require +prefix, parens, or 10+ pure digits to avoid matching
    # numeric ranges like "1967-1975" or "14848-30000"
    'phone': r'^(?:\+[0-9]{1,4}[\s-]?|[(][0-9]{1,4}[)][\s-]?)[0-9][-\s\./0-9]{6,}$|^[0-9]{10,}$',
    'credit_card': r'^[0-9]{13,19}$',
    'ssn_us': r'^\d{3}-\d{2}-\d{4}$',
    'iban': r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{4,}$',
}

QI_KEYWORDS: Dict[str, Any] = {
    'direct_identifiers': [
        'id', 'ssn', 'passport', 'email', 'phone', 'name', 'address',
        'license', 'account', 'card', 'social_security', 'employee_id',
        'patient_id', 'customer_id', 'user_id', 'member_id',
    ],
    'definite_qis': {
        'age': 1.00, 'gender': 1.00, 'sex': 1.00,
        'zipcode': 1.00, 'zip': 1.00, 'postal': 1.00, 'postcode': 1.00,
        'race': 1.00, 'ethnicity': 1.00, 'ethnic': 1.00,
        'occupation': 0.95, 'job': 0.90, 'profession': 0.95,
        'city': 0.95, 'town': 0.90, 'county': 0.95,
        'state': 0.90, 'province': 0.90, 'region': 0.90, 'country': 0.85,
        'nationality': 0.90, 'marital': 0.95, 'education': 0.95,
        'birthdate': 1.00, 'dob': 1.00, 'birth_date': 1.00,
        'birth_year': 0.95, 'birth_month': 0.90,
    },
    'probable_qis': {
        'year': 0.70, 'month': 0.65, 'date': 0.60,
        'income': 0.75, 'salary': 0.75, 'wage': 0.70,
        'religion': 0.80, 'language': 0.70,
        'height': 0.65, 'weight': 0.65,
        'diagnosis': 0.80, 'icd': 0.75,
        'department': 0.60, 'unit': 0.55, 'ward': 0.60, 'specialty': 0.60,
        'employer': 0.70, 'company': 0.65, 'organization': 0.60,
    },
    'possible_qis': {
        'type': 0.40, 'category': 0.45, 'group': 0.40, 'class': 0.45,
        'level': 0.40, 'status': 0.50, 'code': 0.35, 'flag': 0.30,
    },
    'qi_override_patterns': [
        'zip', 'postal', 'diagnosis', 'icd', 'cpt', 'drg',
    ],
}

COLUMN_TYPE_KEYWORDS: Dict[str, List[str]] = {
    'identifier': [
        'id', 'uuid', 'guid', 'key', 'number', 'num', 'no', 'code',
        'ssn', 'ein', 'tin', 'account', 'license', 'permit',
        'index', 'idx', 'pk', 'fk', 'serial',
    ],
    'binary': [
        'flag', 'indicator', 'is_', 'has_', 'was_', 'active',
        'enabled', 'disabled', 'yes', 'no', 'true', 'false',
        'valid', 'invalid', 'status',
    ],
    'sensitive': [
        'disease', 'diagnosis', 'condition', 'symptom', 'treatment',
        'medication', 'drug', 'prescription', 'health', 'medical',
        'income', 'salary', 'wage', 'earnings', 'payment', 'debt',
        'credit', 'score', 'rating', 'criminal', 'arrest', 'conviction',
        'orientation', 'preference', 'political', 'religion', 'belief',
        'pregnancy', 'fertility', 'disability', 'hiv', 'aids', 'mental',
        'psychiatric', 'addiction', 'substance', 'alcohol', 'abuse',
    ],
    'date': [
        'date', 'time', 'datetime', 'timestamp', 'created', 'updated',
        'modified', 'at', 'on', 'when', 'year', 'month', 'day',
        'birth', 'dob', 'start', 'end', 'begin', 'finish',
        '\u03b7\u03bc\u03b5\u03c1\u03bf\u03bc\u03b7\u03bd\u03af\u03b1',
        '\u03b7\u03bc\u03b5\u03c1\u03bf\u03bc\u03b7\u03bd\u03b9\u03b1',
        '\u03b5\u03c4\u03bf\u03c2', '\u03ad\u03c4\u03bf\u03c2',
        '\u03bc\u03ae\u03bd\u03b1\u03c2', '\u03bc\u03b7\u03bd\u03b1\u03c2',
    ],
}

# Greek identifier patterns  (from sdc/sdc_preprocessing.py)
_GREEK_PATTERNS = {
    'afm': (r'^\d{9}$', '\u0391\u03a6\u039c - Tax ID', 0.80),
    'amka': (r'^\d{11}$', '\u0391\u039c\u039a\u0391 - Social Security', 0.80),
    'adt': (r'^[\u0391-\u03a9A-Z]{1,2}\s?\d{6}$',
            '\u0391\u0394\u03a4 - ID Card', 0.70),
    'iban_gr': (r'^GR\d{25}$', 'Greek IBAN', 0.80),
    'vat_gr': (r'^EL\d{9}$', 'Greek VAT Number', 0.80),
}

_GREEK_NAME_HINTS = {
    'afm': ['\u03b1\u03c6\u03bc', 'afm', 'tin', 'tax_id', 'taxid',
            'vat_number'],
    'amka': ['\u03b1\u03bc\u03ba\u03b1', 'amka', 'ssn',
             'social_security', 'insurance_number'],
    'adt': ['\u03b1\u03b4\u03c4', 'adt', 'id_card', 'identity',
            '\u03c4\u03b1\u03c5\u03c4\u03bf\u03c4\u03b7\u03c4\u03b1',
            'tautotita'],
    'iban': ['iban', 'bank_account',
             '\u03c4\u03c1\u03b1\u03c0\u03b5\u03b6\u03b9\u03ba\u03cc\u03c2'],
}


# ============================================================================
# 1. Detailed type  (duplicated from sdc/column_types.py)
# ============================================================================

def _detect_semantic_by_name(var_lower: str) -> str:
    parts = var_lower.split('_')

    for category, keywords in DIRECT_IDENTIFIER_KEYWORDS.items():
        for kw in keywords:
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

    definite_qis = QI_KEYWORDS.get('definite_qis', {})
    for kw in definite_qis:
        if kw in parts or var_lower == kw or var_lower.startswith(kw):
            demo = {'age', 'gender', 'sex', 'race', 'ethnicity', 'ethnic',
                    'marital', 'education', 'nationality', 'occupation',
                    'job', 'profession'}
            geo = {'zipcode', 'zip', 'postal', 'postcode', 'city', 'town',
                   'county', 'state', 'province', 'region', 'country'}
            date = {'birthdate', 'dob', 'birth_date', 'birth_year',
                    'birth_month'}
            if kw in demo:
                return f'{kw.title()} (demographic)'
            elif kw in geo:
                return f'{kw.title()} (geographic)'
            elif kw in date:
                return 'Birth date (temporal ID)'
            return f'{kw.title()} (quasi-identifier)'

    probable_qis = QI_KEYWORDS.get('probable_qis', {})
    for kw in probable_qis:
        if kw in parts or var_lower == kw:
            if kw in {'income', 'salary', 'wage'}:
                return 'Income/Financial'
            elif kw in {'diagnosis', 'icd'}:
                return 'Medical (sensitive)'
            return f'{kw.title()}'

    for category, keywords in COLUMN_TYPE_KEYWORDS.items():
        for kw in keywords:
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
    sample = col.dropna().astype(str)
    if len(sample) == 0:
        return ''
    sample = sample.head(200)

    best_match, best_pct = '', 0.0
    for pattern_name, pattern in DIRECT_IDENTIFIER_PATTERNS.items():
        try:
            pct = sample.str.match(pattern, na=False).mean()
            if pct > 0.5 and pct > best_pct:
                best_pct = pct
                best_match = {
                    'email': 'Email', 'phone': 'Phone',
                    'credit_card': 'Credit Card', 'ssn_us': 'SSN (US)',
                    'iban': 'IBAN',
                }.get(pattern_name, pattern_name)
        except Exception:
            continue
    if best_match:
        return best_match

    addr_kws = {'street', 'st', 'ave', 'avenue', 'road', 'rd', 'blvd',
                'drive', 'dr', 'lane', 'ln', 'way', 'court', 'ct',
                'place', 'pl', 'circle', 'terrace', 'highway', 'hwy'}
    if sample.str.lower().apply(
        lambda x: any(k in str(x).split() for k in addr_kws)
    ).mean() > 0.3:
        return 'Address'

    if sample.str.match(r'^\d{5}(-\d{4})?$', na=False).mean() > 0.5:
        return 'Zipcode (geographic)'

    for dp in [r'^\d{4}-\d{2}-\d{2}', r'^\d{2}/\d{2}/\d{4}',
               r'^\d{2}-\d{2}-\d{4}']:
        if sample.str.match(dp, na=False).mean() > 0.5:
            return 'Date string'
    return ''


def _classify_integer(col, nunique, n_rows, tag):
    non_null = col.dropna()
    is_ord = False
    if nunique <= 30 and len(non_null) > 0:
        vals = sorted(non_null.unique())
        if vals[0] in (0, 1) and vals[-1] == vals[0] + nunique - 1:
            is_ord = True
    if tag:
        if nunique <= 2:
            return f'Integer -- Binary ({tag})'
        elif is_ord:
            return f'Integer -- Ordinal ({nunique} levels, {tag})'
        elif nunique <= 20:
            return f'Integer -- Categorical ({tag})'
        elif nunique > n_rows * 0.5:
            return f'Integer -- Identifier ({tag})'
        else:
            return f'Integer -- Continuous range ({tag})'
    if nunique <= 2:
        return 'Integer -- Binary'
    elif is_ord and nunique <= 10:
        return f'Integer -- Ordinal ({nunique} levels)'
    elif is_ord:
        return (f'Integer -- Ordinal ({nunique} levels, '
                f'range {int(col.min())}-{int(col.max())})')
    elif nunique <= 20:
        return f'Integer -- Categorical ({nunique} values)'
    elif nunique > n_rows * 0.5:
        return 'Integer -- Identifier-like (high cardinality)'
    else:
        return (f'Integer -- Continuous '
                f'(range {int(col.min())}-{int(col.max())})')


def _classify_float(col, nunique, n_rows, tag):
    if tag:
        if nunique <= 10:
            return f'Float -- Categorical ({tag})'
        return f'Float -- Continuous ({tag})'
    if nunique <= 10:
        return f'Float -- Categorical ({nunique} values)'
    try:
        rng = f'{col.min():.1f}-{col.max():.1f}'
    except Exception:
        rng = 'range'
    return f'Float -- Continuous ({rng})'


def _classify_string(col, nunique, n_rows, tag):
    sample = col.dropna().head(200)
    sample = sample[sample.astype(str).str.strip() != '']
    numeric_pct = 0
    if len(sample) > 0:
        numeric_pct = pd.to_numeric(sample, errors='coerce').notna().mean()
    if numeric_pct <= 0.8 and len(sample) > 0:
        if sample.astype(str).str.contains(',').mean() > 0.3:
            eu = (sample.astype(str)
                  .str.replace('.', '', regex=False)
                  .str.replace(',', '.', regex=False))
            numeric_pct = pd.to_numeric(eu, errors='coerce').notna().mean()
    if numeric_pct > 0.8:
        if tag:
            if nunique <= 20:
                return f'Char (numeric) -- Coded ({tag})'
            return f'Char (numeric) -- Continuous ({tag})'
        if nunique <= 20:
            return f'Char (numeric) -- Coded ({nunique} values)'
        return 'Char (numeric) -- Continuous'
    avg_tok = 1.0
    if len(sample) > 0:
        avg_tok = sample.astype(str).str.split().str.len().mean()
    if tag:
        hint = ''
        if nunique <= 2:
            hint = ', binary'
        elif nunique <= 20:
            hint = f', {nunique} values'
        elif nunique > n_rows * 0.5:
            hint = ', high cardinality'
        return f'Character -- {tag}{hint}'
    if nunique <= 2:
        return 'Character -- Binary (2 values)'
    elif nunique <= 10:
        return f'Character -- Categorical ({nunique} levels)'
    elif nunique <= 20:
        return f'Character -- Categorical ({nunique} values)'
    elif nunique > n_rows * 0.5:
        if avg_tok > 3:
            return 'Character -- Free text (high cardinality)'
        return 'Character -- High-cardinality (ID-like)'
    elif avg_tok > 3:
        return f'Character -- Free text ({nunique} unique)'
    return f'Character -- Categorical ({nunique} values, high-card)'


def classify_column_type(var_name, col, nunique, n_rows):
    """Classify a column into a detailed semantic + structural type."""
    var_lower = var_name.lower().replace(' ', '_').replace('-', '_')
    tag = _detect_semantic_by_name(var_lower)
    if not tag and (col.dtype == 'object'
                    or pd.api.types.is_string_dtype(col)):
        tag = _detect_semantic_by_values(col)
    if pd.api.types.is_datetime64_any_dtype(col):
        base = 'Date -- Temporal'
        return f'{base} ({tag})' if tag else base
    elif pd.api.types.is_bool_dtype(col):
        base = 'Boolean -- Binary'
        return f'{base} ({tag})' if tag else base
    elif pd.api.types.is_integer_dtype(col):
        return _classify_integer(col, nunique, n_rows, tag)
    elif pd.api.types.is_float_dtype(col):
        return _classify_float(col, nunique, n_rows, tag)
    elif col.dtype == 'object' or pd.api.types.is_string_dtype(col):
        return _classify_string(col, nunique, n_rows, tag)
    return str(col.dtype)


def classify_columns(df, columns=None):
    """Classify all (or selected) columns. Returns {col: label}."""
    cols = columns or list(df.columns)
    n_rows = len(df)
    return {c: classify_column_type(c, df[c], df[c].nunique(), n_rows)
            for c in cols if c in df.columns}


# ============================================================================
# 2. Structural type  (duplicated from sdc/detection/column_types.py)
# ============================================================================

def _is_sequential_id(series):
    try:
        if not (series % 1 == 0).all():
            return False
        # Low-cardinality integers (≤30 unique) are ordinal/categorical, not IDs
        if series.nunique() <= 30:
            return False
        min_val = series.min()
        if min_val > 10:
            return False
        sv = series.sort_values().values
        if len(sv) < 2:
            return False
        return np.median(np.diff(sv)) <= 1.5 and min_val >= 0
    except Exception:
        return False


def identify_column_types(data):
    """Simple structural type: continuous / categorical / identifier / binary."""
    id_kws = COLUMN_TYPE_KEYWORDS.get('identifier', [])
    val_kws = [
        'income', 'salary', 'wage', 'price', 'cost', 'amount',
        'score', 'rating', 'value', 'total', 'sum', 'count',
        'weight', 'height', 'age', 'year', 'date', 'time',
        'balance', 'revenue', 'profit', 'loss', 'tax', 'fee',
    ]
    out = {}
    for col in data.columns:
        cl = col.lower()
        is_id = any(k in cl for k in id_kws)
        is_val = any(k in cl for k in val_kws)
        if pd.api.types.is_numeric_dtype(data[col]):
            nu = data[col].nunique()
            if 'age' in cl and pd.api.types.is_integer_dtype(data[col]):
                out[col] = 'categorical'; continue
            if nu <= 2:
                out[col] = 'binary'
            elif is_id and not is_val:
                out[col] = 'identifier'
            elif is_val:
                out[col] = 'continuous'
            elif _is_sequential_id(data[col]):
                out[col] = 'identifier'
            elif nu < 20 and (data[col] % 1 == 0).all():
                out[col] = 'categorical'
            else:
                out[col] = 'continuous'
        else:
            ur = data[col].nunique() / len(data) if len(data) > 0 else 0
            if is_id and not is_val:
                out[col] = 'identifier'
            elif ur > 0.95:
                out[col] = 'identifier'
            else:
                out[col] = 'categorical'
    return out


# ============================================================================
# 3. Direct-identifier detection  (duplicated from sdc/detection/column_types.py)
# ============================================================================

def auto_detect_direct_identifiers(data, check_patterns=True):
    """Detect columns containing direct identifiers (name, email, SSN, etc.)."""
    det = {}
    for col in data.columns:
        cl = col.lower().replace(' ', '_').replace('-', '_')
        words = cl.split('_')
        for id_type, kws in DIRECT_IDENTIFIER_KEYWORDS.items():
            if any(k in words or k == cl for k in kws):
                det[col] = id_type; break
        if col not in det and check_patterns and data[col].dtype == 'object':
            sample = data[col].dropna().head(100).astype(str)
            if len(sample) > 0:
                for pn, pat in DIRECT_IDENTIFIER_PATTERNS.items():
                    try:
                        if sample.str.match(pat, na=False).sum() / len(sample) > 0.5:
                            det[col] = pn; break
                    except Exception:
                        pass
    return det


# ============================================================================
# 4. Greek identifiers  (duplicated from sdc/sdc_preprocessing.py)
# ============================================================================

def detect_greek_identifiers(data):
    """Detect Greek-specific direct identifiers (\u0391\u03a6\u039c, \u0391\u039c\u039a\u0391, \u0391\u0394\u03a4, IBAN, VAT)."""
    det = {}
    for col in data.columns:
        cl = col.lower().replace(' ', '_').replace('-', '_')
        words = cl.split('_')
        for id_type, hints in _GREEK_NAME_HINTS.items():
            if any(h in words or h == cl for h in hints):
                desc = _GREEK_PATTERNS.get(id_type, ('', id_type))[1]
                det[col] = f'{desc} (name match)'; break
        if col in det:
            continue
        if data[col].dtype == 'object':
            sample = data[col].dropna().astype(str).head(200)
            if len(sample) == 0:
                continue
            for pn, (pat, desc, minr) in _GREEK_PATTERNS.items():
                try:
                    mr = sample.str.match(pat, na=False).mean()
                    if mr >= minr:
                        det[col] = f'{desc} (pattern: {mr:.0%})'; break
                except re.error:
                    continue
    return det


# ============================================================================
# 5. Orchestrator
# ============================================================================

def classify_all(data, sample_n=None):
    """Classify all columns. Returns list of dicts."""
    if sample_n and len(data) > sample_n:
        data = data.sample(n=sample_n, random_state=42)
    n_rows = len(data)

    detailed = classify_columns(data)
    structural = identify_column_types(data)
    direct_ids = auto_detect_direct_identifiers(data, check_patterns=True)
    greek_ids = detect_greek_identifiers(data)

    results = []
    for col in data.columns:
        s = data[col]
        nu = s.nunique()
        n_na = int(s.isna().sum())
        miss = n_na / n_rows if n_rows > 0 else 0
        vals = s.dropna().unique()[:5].tolist()

        # Numeric stats (try coercing string-stored numerics too)
        num = pd.to_numeric(s, errors='coerce') if s.dtype == 'object' else s
        is_num = pd.api.types.is_numeric_dtype(num) and num.notna().sum() > 0
        mean_val = round(float(num.mean()), 4) if is_num else None
        sd_val = round(float(num.std()), 4) if is_num else None
        min_val = float(num.min()) if is_num else None
        max_val = float(num.max()) if is_num else None

        # Skewness (numeric only, useful for top/bottom coding decisions)
        skew_val = None
        if is_num and num.notna().sum() > 10:
            try:
                skew_val = round(float(num.skew()), 4)
            except Exception:
                pass

        # Categorical stats: most frequent value and its count/pct
        top_value, top_count, top_pct = None, None, None
        vc = s.value_counts(dropna=True)
        if len(vc) > 0:
            top_value = str(vc.index[0])
            top_count = int(vc.iloc[0])
            top_pct = round(top_count / n_rows * 100, 1) if n_rows > 0 else 0

        # SDC-relevant risk indicators
        uniqueness_ratio = round(nu / n_rows, 4) if n_rows > 0 else 0
        near_constant = top_pct is not None and top_pct > 95.0

        # Shannon entropy (categorical) — higher = more uniform = harder to anonymise
        entropy_val = None
        if len(vc) > 1:
            probs = vc.values / vc.values.sum()
            entropy_val = round(float(-np.sum(probs * np.log2(probs))), 4)

        # SDC flag: column likely problematic for anonymisation
        sdc_flag = ''
        if uniqueness_ratio > 0.95 and not (col in direct_ids):
            sdc_flag = 'QUASI-ID (near-unique)'
        elif near_constant:
            sdc_flag = 'NEAR-CONSTANT (>95% one value)'
        elif miss > 0.5:
            sdc_flag = 'HIGH-MISSING (>50% null)'
        elif is_num and skew_val is not None and abs(skew_val) > 2.0:
            sdc_flag = 'SKEWED (needs top/bottom coding)'

        results.append({
            'column': col,
            'dtype': str(s.dtype),
            'n_unique': nu,
            'n_rows': n_rows,
            'n_na': n_na,
            'missing_pct': round(miss * 100, 1),
            'uniqueness_ratio': uniqueness_ratio,
            'mean': mean_val,
            'std': sd_val,
            'min': min_val,
            'max': max_val,
            'skewness': skew_val,
            'top_value': top_value,
            'top_count': top_count,
            'top_pct': top_pct,
            'near_constant': near_constant,
            'entropy': entropy_val,
            'sdc_flag': sdc_flag,
            'detailed_type': detailed.get(col, '?'),
            'structural_type': structural.get(col, '?'),
            'is_direct_identifier': col in direct_ids,
            'identifier_type': direct_ids.get(col, ''),
            'is_greek_id': col in greek_ids,
            'greek_id_type': greek_ids.get(col, ''),
            'sample_values': ', '.join(str(v) for v in vals),
        })
    return results


# ============================================================================
# 6. Display
# ============================================================================

def _c(text, code):
    m = {'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m',
         'cyan': '\033[96m', 'magenta': '\033[95m', 'grey': '\033[90m',
         'bold': '\033[1m', 'reset': '\033[0m'}
    return f"{m.get(code, '')}{text}{m['reset']}"


def print_report(results, file_path):
    n = len(results)
    nid = sum(1 for r in results if r['is_direct_identifier'] or r['is_greek_id'])
    nc = sum(1 for r in results if r['structural_type'] == 'continuous')
    ncat = sum(1 for r in results if r['structural_type'] == 'categorical')
    nb = sum(1 for r in results if r['structural_type'] == 'binary')
    ni = sum(1 for r in results if r['structural_type'] == 'identifier')
    nr = results[0]['n_rows'] if results else 0

    print(f"\n{'='*90}")
    print(f"  Column Classification Report -- {Path(file_path).name}  ({nr:,} rows)")
    print(f"{'='*90}")
    print(f"  {n} columns | {nid} direct identifiers | "
          f"{nc} continuous | {ncat} categorical | {nb} binary | {ni} identifier-like")
    print(f"{'='*90}\n")

    ids = [r for r in results if r['is_direct_identifier'] or r['is_greek_id']]
    if ids:
        print(_c("  DIRECT IDENTIFIERS (remove before anonymisation):", 'red'))
        print(f"  {'-'*86}")
        for r in ids:
            lbl = r['identifier_type'] or r['greek_id_type']
            print(f"  {_c('!', 'red')}  {_c(r['column'], 'bold'):40s}  "
                  f"{_c(lbl, 'red'):30s}  {r['detailed_type']}")
        print()

    # --- Main table: type + cardinality + NA ---
    print(f"  {'Column':<28s}  {'Structural':<14s}  {'Detailed Type':<42s}  "
          f"{'Unique':>7s}  {'NA':>6s}  {'Miss%':>6s}")
    print(f"  {'-'*28}  {'-'*14}  {'-'*42}  "
          f"{'-'*7}  {'-'*6}  {'-'*6}")

    for r in results:
        cn = r['column'][:28]
        st = r['structural_type']
        sc = {'identifier': 'red', 'continuous': 'cyan',
              'binary': 'magenta'}.get(st, 'green')
        ms = f"{r['missing_pct']:.1f}%" if r['missing_pct'] > 0 else ''
        na = f"{r['n_na']:,d}" if r['n_na'] > 0 else ''
        flag = r['is_direct_identifier'] or r['is_greek_id']
        pre = _c(f"! {cn}", 'red') if flag else f"  {cn}"
        pad = 38 if flag else 30
        dt = r['detailed_type'][:42]
        print(f"  {pre:<{pad}s}  {_c(f'{st:<14s}', sc)}  {dt:<42s}  "
              f"{r['n_unique']:>7,d}  {na:>6s}  {ms:>6s}")

    # --- Numeric stats ---
    numerics = [r for r in results if r['mean'] is not None]
    if numerics:
        print(f"\n  {'-'*96}")
        print(f"  Numeric column statistics:")
        print(f"  {'Column':<28s}  {'Mean':>14s}  {'Std':>14s}  "
              f"{'Min':>14s}  {'Max':>14s}  {'Skew':>8s}")
        print(f"  {'-'*28}  {'-'*14}  {'-'*14}  "
              f"{'-'*14}  {'-'*14}  {'-'*8}")
        def _fmt(v):
            if v is None:
                return ''
            if abs(v) >= 1000:
                return f"{v:>14,.2f}"
            return f"{v:>14.4f}"
        for r in numerics:
            sk = ''
            if r['skewness'] is not None:
                sk = f"{r['skewness']:>8.2f}"
                if abs(r['skewness']) > 2.0:
                    sk = _c(sk, 'yellow')
            print(f"  {r['column'][:28]:<28s}  {_fmt(r['mean'])}  {_fmt(r['std'])}  "
                  f"{_fmt(r['min'])}  {_fmt(r['max'])}  {sk}")

    # --- Categorical stats: top value + entropy ---
    cats = [r for r in results if r['top_value'] is not None]
    if cats:
        print(f"\n  {'-'*96}")
        print(f"  Top category per column:")
        print(f"  {'Column':<28s}  {'Top Value':<30s}  {'Count':>8s}  "
              f"{'Freq%':>6s}  {'Entropy':>8s}  {'Uniq%':>6s}")
        print(f"  {'-'*28}  {'-'*30}  {'-'*8}  "
              f"{'-'*6}  {'-'*8}  {'-'*6}")
        for r in cats:
            tv = (r['top_value'] or '')[:30]
            tc = f"{r['top_count']:>8,d}" if r['top_count'] else ''
            tp = f"{r['top_pct']:.1f}%" if r['top_pct'] else ''
            ent = f"{r['entropy']:.2f}" if r['entropy'] is not None else ''
            uq = f"{r['uniqueness_ratio']*100:.1f}%"
            print(f"  {r['column'][:28]:<28s}  {tv:<30s}  {tc}  "
                  f"{tp:>6s}  {ent:>8s}  {uq:>6s}")

    # --- SDC Risk Assessment ---
    flagged = [r for r in results
               if r['sdc_flag'] or r['is_direct_identifier'] or r['is_greek_id']]
    if flagged:
        print(f"\n  {'-'*96}")
        print(_c("  SDC Risk Assessment:", 'bold'))
        print(f"  {'-'*96}")
        for r in flagged:
            col_name = r['column'][:28]
            if r['is_direct_identifier'] or r['is_greek_id']:
                lbl = r['identifier_type'] or r['greek_id_type']
                print(f"  {_c('!', 'red')}  {col_name:<28s}  "
                      f"{_c('DIRECT IDENTIFIER', 'red')}  ({lbl})")
            elif r['sdc_flag']:
                color = 'yellow' if 'SKEWED' in r['sdc_flag'] else 'red'
                print(f"  {_c('*', color)}  {col_name:<28s}  "
                      f"{_c(r['sdc_flag'], color)}")
        # Summary
        n_near_unique = sum(1 for r in results
                            if r['uniqueness_ratio'] > 0.5
                            and r['structural_type'] != 'identifier'
                            and not r['is_direct_identifier'])
        n_near_const = sum(1 for r in results if r['near_constant'])
        n_skewed = sum(1 for r in results
                       if r['skewness'] is not None and abs(r['skewness']) > 2.0)
        n_high_miss = sum(1 for r in results if r['missing_pct'] > 50)
        print(f"\n  Summary: {nid} identifiers to remove, "
              f"{n_near_unique} near-unique (QI candidates), "
              f"{n_near_const} near-constant (exclude), "
              f"{n_skewed} skewed (top/bottom code), "
              f"{n_high_miss} high-missing (>50%)")

    # --- Sample values ---
    print(f"\n  {'-'*96}")
    print(f"  Sample values (first 5 unique non-null):")
    print(f"  {'-'*96}")
    for r in results:
        v = r['sample_values']
        if len(v) > 75:
            v = v[:72] + '...'
        print(f"  {r['column'][:28]:<28s}  {_c(v, 'grey')}")
    print()


# ============================================================================
# 7. File loading
# ============================================================================

def load_data(file_path, sheet=None):
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext == '.csv':
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(file_path, sep=sep, nrows=5)
                if len(df.columns) > 1:
                    return pd.read_csv(file_path, sep=sep)
            except Exception:
                continue
        return pd.read_csv(file_path)
    elif ext in ('.xlsx', '.xls', '.xlsm'):
        kw = {'sheet_name': sheet} if sheet else {}
        return pd.read_excel(file_path, **kw)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    elif ext == '.tsv':
        return pd.read_csv(file_path, sep='\t')
    raise ValueError(f"Unsupported format: {ext}. Use CSV, Excel, Parquet, or TSV.")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Classify column types and detect direct identifiers.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify_columns.py data.csv
  python classify_columns.py data.xlsx --sheet "Sheet2"
  python classify_columns.py data.parquet --sample 5000
  python classify_columns.py data.csv --json
  python classify_columns.py data.csv --out report.csv
""")
    ap.add_argument('file', help='Path to data file (CSV, Excel, Parquet, TSV)')
    ap.add_argument('--sheet', help='Sheet name for Excel files')
    ap.add_argument('--sample', type=int, help='Sample N rows for large files')
    ap.add_argument('--json', action='store_true', help='Output as JSON')
    ap.add_argument('--out', help='Save report to CSV file')
    args = ap.parse_args()

    try:
        data = load_data(args.file, sheet=args.sheet)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data):,} rows x {len(data.columns)} columns from {args.file}")
    results = classify_all(data, sample_n=args.sample)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif args.out:
        pd.DataFrame(results).to_csv(args.out, index=False)
        print(f"Report saved to {args.out}")
    else:
        print_report(results, args.file)


if __name__ == '__main__':
    main()
