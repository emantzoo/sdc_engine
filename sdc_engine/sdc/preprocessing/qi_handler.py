"""
QI Handler - High Cardinality Quasi-Identifier Preprocessing
=============================================================

Handles high-cardinality QIs that make k-anonymity infeasible.
Provides automatic and user-defined strategies for reducing QI combination space.

Decision Framework:
1. Option A: Hierarchical Generalization (PREFERRED) - use coarser level
2. Option B: Reclassify as Sensitive - remove from QI set
3. Option C: Exclude Entirely - drop from release
4. Option D: Top-K + Other - bin rare values

Usage:
    from src.preprocessing.qi_handler import QIHandler

    handler = QIHandler(access_tier='PUBLIC')

    # Analyze QIs
    analysis = handler.analyze_qis(data, quasi_identifiers)

    # Apply preprocessing with user-defined hierarchies
    processed_data, processed_qis = handler.preprocess_qis(
        data,
        quasi_identifiers,
        hierarchies={
            'municipality': {'target': 'region', 'mapping': muni_to_region},
            'occupation_code': {'strategy': 'top_k', 'k': 30},
        }
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum


class AccessTier(Enum):
    """Access tier definitions with cardinality constraints."""
    PUBLIC = 'PUBLIC'
    SCIENTIFIC = 'SCIENTIFIC'
    SECURE = 'SECURE'


# Cardinality thresholds by access tier
TIER_CONSTRAINTS = {
    AccessTier.PUBLIC: {
        'max_cardinality_per_qi': 50,
        'max_qi_count': 5,
        'max_combination_space': 50000,
        'description': 'Public release files - strictest constraints'
    },
    AccessTier.SCIENTIFIC: {
        'max_cardinality_per_qi': 100,
        'max_qi_count': 6,
        'max_combination_space': 500000,
        'description': 'Scientific use files - moderate constraints'
    },
    AccessTier.SECURE: {
        'max_cardinality_per_qi': 200,
        'max_qi_count': 8,
        'max_combination_space': 2000000,
        'description': 'Secure environment only - relaxed constraints'
    },
}


class PreprocessingStrategy(Enum):
    """Available preprocessing strategies for high-cardinality QIs."""
    HIERARCHY = 'hierarchy'      # Option A: Apply hierarchical generalization
    RECLASSIFY = 'reclassify'    # Option B: Move to sensitive attributes
    EXCLUDE = 'exclude'          # Option C: Drop from dataset
    TOP_K = 'top_k'              # Option D: Keep top K, bin rest as "Other"
    BINNING = 'binning'          # Numeric: equal-width or equal-frequency bins
    KEEP = 'keep'                # No change needed


@dataclass
class QIAnalysis:
    """Analysis result for a single QI."""
    column: str
    cardinality: int
    dtype: str
    is_numeric: bool
    exceeds_threshold: bool
    recommended_strategy: PreprocessingStrategy
    recommendation_reason: str
    value_distribution: Dict[str, int] = field(default_factory=dict)
    top_values_coverage: float = 0.0  # % of records covered by top 10 values
    has_natural_hierarchy: bool = False
    suggested_hierarchy_levels: List[str] = field(default_factory=list)


@dataclass
class PreprocessingPlan:
    """Complete preprocessing plan for all QIs."""
    access_tier: AccessTier
    original_qis: List[str]
    processed_qis: List[str]
    sensitive_vars: List[str]
    excluded_vars: List[str]
    transformations: Dict[str, Dict]
    feasibility_before: Dict
    feasibility_after: Dict
    warnings: List[str]


# Common hierarchies for Greek/European public sector data
COMMON_HIERARCHIES = {
    # Geographic hierarchies
    'municipality': ['prefecture', 'region', 'country'],
    'prefecture': ['region', 'country'],
    'postal_code': ['postal_3digit', 'municipality', 'prefecture'],
    'nuts3': ['nuts2', 'nuts1', 'country'],

    # Occupation hierarchies (ISCO-08)
    'occupation_code': ['occupation_subgroup', 'occupation_group', 'occupation_major'],
    'isco_4digit': ['isco_3digit', 'isco_2digit', 'isco_1digit'],

    # Industry hierarchies (NACE)
    'nace_4digit': ['nace_3digit', 'nace_2digit', 'nace_section'],
    'industry_code': ['industry_group', 'industry_division', 'industry_section'],

    # Education hierarchies (ISCED)
    'education_detail': ['education_level', 'education_broad'],
    'isced_3digit': ['isced_2digit', 'isced_1digit'],

    # Time hierarchies
    'date': ['year_month', 'year_quarter', 'year'],
    'timestamp': ['date', 'year_month', 'year'],
    'age': ['age_5year', 'age_10year', 'age_group'],

    # Income hierarchies
    'income': ['income_decile', 'income_quintile', 'income_bracket'],
    'salary': ['salary_band', 'salary_bracket'],
}


class QIHandler:
    """
    Handles high-cardinality quasi-identifiers through preprocessing.

    Parameters:
    -----------
    access_tier : str or AccessTier
        Target access tier ('PUBLIC', 'SCIENTIFIC', 'SECURE')
    custom_thresholds : dict, optional
        Override default thresholds
    """

    def __init__(
        self,
        access_tier: Union[str, AccessTier] = 'SCIENTIFIC',
        custom_thresholds: Optional[Dict] = None
    ):
        if isinstance(access_tier, str):
            access_tier = AccessTier(access_tier.upper())
        self.access_tier = access_tier

        # Get constraints for this tier
        self.constraints = TIER_CONSTRAINTS[access_tier].copy()
        if custom_thresholds:
            self.constraints.update(custom_thresholds)

    def suggest_column_classification(
        self,
        data: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Auto-detect and suggest column classifications for anonymization.

        Analyzes all columns and suggests:
        - quasi_identifiers: Columns that could identify individuals
        - sensitive: Confidential data that needs protection (but not identifying alone)
        - direct_identifiers: Columns that must be removed (names, IDs, etc.)
        - safe_to_publish: Columns with no disclosure risk
        - needs_preprocessing: High-cardinality columns that need aggregation

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        verbose : bool
            Print suggestions

        Returns:
        --------
        dict : {
            'quasi_identifiers': [(col, cardinality, reason), ...],
            'sensitive': [(col, reason), ...],
            'direct_identifiers': [(col, reason), ...],
            'safe_to_publish': [(col, reason), ...],
            'needs_preprocessing': [(col, cardinality, suggested_strategy), ...],
            'warnings': [str, ...]
        }
        """
        result = {
            'quasi_identifiers': [],
            'sensitive': [],
            'direct_identifiers': [],
            'safe_to_publish': [],
            'needs_preprocessing': [],
            'warnings': [],
        }

        n_records = len(data)
        max_card = self.constraints['max_cardinality_per_qi']

        # Patterns for detection
        direct_id_patterns = ['id', 'ssn', 'social_security', 'tax_id', 'afm', 'amka',
                              'passport', 'license', 'phone', 'email', 'address',
                              'name', 'surname', 'firstname', 'lastname']

        sensitive_patterns = ['salary', 'income', 'wage', 'earnings', 'diagnosis',
                              'disease', 'condition', 'treatment', 'score', 'grade',
                              'debt', 'loan', 'credit', 'tax_amount', 'benefit']

        qi_patterns = ['age', 'sex', 'gender', 'birth', 'occupation', 'profession',
                       'education', 'municipality', 'region', 'postal', 'zip',
                       'marital', 'nationality', 'ethnicity', 'religion']

        for col in data.columns:
            col_lower = col.lower()
            cardinality = data[col].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(data[col])
            uniqueness = cardinality / n_records if n_records > 0 else 0

            # Check for direct identifiers (MUST REMOVE)
            if any(p in col_lower for p in direct_id_patterns) or uniqueness > 0.9:
                if uniqueness > 0.9:
                    reason = f"Near-unique ({uniqueness:.0%} unique) - likely an identifier"
                else:
                    reason = f"Name pattern suggests direct identifier"
                result['direct_identifiers'].append((col, reason))
                continue

            # Check for sensitive attributes (confidential but not identifying)
            if any(p in col_lower for p in sensitive_patterns):
                reason = f"Contains sensitive information ({col_lower})"
                result['sensitive'].append((col, reason))
                continue

            # Check for quasi-identifiers
            is_qi = False
            qi_reason = ""

            if any(p in col_lower for p in qi_patterns):
                is_qi = True
                qi_reason = f"Common QI pattern ({col_lower})"
            elif cardinality <= 100 and cardinality >= 2:
                # Categorical with moderate cardinality = potential QI
                is_qi = True
                qi_reason = f"Categorical ({cardinality} values) - may identify in combination"
            elif is_numeric and 'year' in col_lower:
                is_qi = True
                qi_reason = f"Year variable - temporal QI"

            if is_qi:
                result['quasi_identifiers'].append((col, cardinality, qi_reason))

                # Check if needs preprocessing
                if cardinality > max_card:
                    if any(h in col_lower for h in ['municipality', 'postal', 'occupation', 'code']):
                        strategy = 'hierarchy'
                    elif is_numeric:
                        strategy = 'binning'
                    else:
                        strategy = 'top_k'
                    result['needs_preprocessing'].append((col, cardinality, strategy))
                continue

            # Safe to publish (low risk)
            if cardinality < 2:
                result['safe_to_publish'].append((col, "Constant value"))
            elif is_numeric and cardinality > 100:
                result['safe_to_publish'].append((col, "High-cardinality numeric (aggregate only)"))

        # Warnings
        if len(result['quasi_identifiers']) > self.constraints['max_qi_count']:
            result['warnings'].append(
                f"Found {len(result['quasi_identifiers'])} potential QIs, "
                f"but max for {self.access_tier.value} is {self.constraints['max_qi_count']}. "
                f"Consider excluding some."
            )

        if len(result['needs_preprocessing']) > 0:
            result['warnings'].append(
                f"{len(result['needs_preprocessing'])} QIs exceed cardinality threshold ({max_card}) "
                f"and need preprocessing before anonymization."
            )

        if verbose:
            print("\n" + "=" * 70)
            print(f"  COLUMN CLASSIFICATION SUGGESTIONS - {self.access_tier.value}")
            print("=" * 70)

            if result['direct_identifiers']:
                print("\n  [!] DIRECT IDENTIFIERS (MUST REMOVE):")
                for col, reason in result['direct_identifiers']:
                    print(f"      - {col}: {reason}")

            if result['sensitive']:
                print("\n  [S] SENSITIVE ATTRIBUTES (confidential, not identifying):")
                for col, reason in result['sensitive']:
                    print(f"      - {col}: {reason}")

            if result['quasi_identifiers']:
                print("\n  [Q] QUASI-IDENTIFIERS (use for k-anonymity):")
                for col, card, reason in result['quasi_identifiers']:
                    needs_pp = " [NEEDS PREPROCESSING]" if card > max_card else ""
                    print(f"      - {col}: {card} unique{needs_pp}")
                    print(f"        {reason}")

            if result['needs_preprocessing']:
                print("\n  [P] NEED PREPROCESSING (cardinality > {max_card}):")
                for col, card, strategy in result['needs_preprocessing']:
                    print(f"      - {col}: {card} unique -> suggest '{strategy}'")

            if result['safe_to_publish']:
                print("\n  [OK] SAFE TO PUBLISH (low risk):")
                for col, reason in result['safe_to_publish'][:5]:
                    print(f"      - {col}: {reason}")
                if len(result['safe_to_publish']) > 5:
                    print(f"      ... and {len(result['safe_to_publish']) - 5} more")

            if result['warnings']:
                print("\n  [!] WARNINGS:")
                for w in result['warnings']:
                    print(f"      - {w}")

            # Calculate feasibility if we have QIs
            if result['quasi_identifiers']:
                qis = [q[0] for q in result['quasi_identifiers']]
                feasibility = self.calculate_feasibility(data, qis)
                print(f"\n  FEASIBILITY (with all {len(qis)} QIs):")
                print(f"      Combination space: {feasibility['combination_space']:,}")
                print(f"      Expected EQ size: {feasibility['expected_eq_size']:.2f}")
                print(f"      Feasibility: {feasibility['feasibility'].upper()}")
                print(f"      Max achievable k: {feasibility['max_achievable_k']}")

        return result

    def analyze_qis(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_vars: Optional[List[str]] = None
    ) -> Dict[str, QIAnalysis]:
        """
        Analyze quasi-identifiers and recommend preprocessing strategies.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : list
            List of QI column names
        sensitive_vars : list, optional
            Already-identified sensitive variables

        Returns:
        --------
        dict : Column name -> QIAnalysis
        """
        sensitive_vars = sensitive_vars or []
        analyses = {}

        max_card = self.constraints['max_cardinality_per_qi']

        for qi in quasi_identifiers:
            if qi not in data.columns:
                continue

            col = data[qi]
            cardinality = col.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(col)
            exceeds = cardinality > max_card

            # Calculate value distribution
            value_counts = col.value_counts()
            top_10_coverage = value_counts.head(10).sum() / len(data) if len(data) > 0 else 0

            # Determine recommended strategy
            strategy, reason = self._recommend_strategy(
                qi, cardinality, is_numeric, top_10_coverage, max_card
            )

            # Check for natural hierarchies
            has_hierarchy = qi.lower() in COMMON_HIERARCHIES or any(
                h in qi.lower() for h in ['code', 'id', 'postal', 'municipality', 'occupation']
            )

            suggested_levels = COMMON_HIERARCHIES.get(qi.lower(), [])

            analyses[qi] = QIAnalysis(
                column=qi,
                cardinality=cardinality,
                dtype=str(col.dtype),
                is_numeric=is_numeric,
                exceeds_threshold=exceeds,
                recommended_strategy=strategy,
                recommendation_reason=reason,
                value_distribution=value_counts.head(20).to_dict(),
                top_values_coverage=top_10_coverage,
                has_natural_hierarchy=has_hierarchy,
                suggested_hierarchy_levels=suggested_levels
            )

        return analyses

    def _recommend_strategy(
        self,
        column: str,
        cardinality: int,
        is_numeric: bool,
        top_coverage: float,
        max_card: int
    ) -> Tuple[PreprocessingStrategy, str]:
        """Recommend preprocessing strategy based on column characteristics."""

        if cardinality <= max_card:
            return PreprocessingStrategy.KEEP, f"Cardinality {cardinality} within threshold {max_card}"

        col_lower = column.lower()

        # Check for hierarchical columns
        if any(h in col_lower for h in ['municipality', 'postal', 'occupation', 'nace', 'isco', 'code']):
            return PreprocessingStrategy.HIERARCHY, \
                f"High cardinality ({cardinality}) but has natural hierarchy - generalize to coarser level"

        # Numeric columns - use binning
        if is_numeric:
            return PreprocessingStrategy.BINNING, \
                f"Numeric with high cardinality ({cardinality}) - apply binning"

        # Long-tail distribution - top K + Other
        if top_coverage >= 0.7:
            return PreprocessingStrategy.TOP_K, \
                f"Top 10 values cover {top_coverage:.0%} - use Top-K with 'Other' category"

        # Check if it might be a sensitive attribute
        if any(s in col_lower for s in ['salary', 'income', 'diagnosis', 'disease', 'amount', 'score']):
            return PreprocessingStrategy.RECLASSIFY, \
                f"High cardinality ({cardinality}) and appears to be sensitive - reclassify as sensitive attribute"

        # Check if likely an ID or free text
        if any(s in col_lower for s in ['id', 'name', 'text', 'comment', 'description', 'address']):
            return PreprocessingStrategy.EXCLUDE, \
                f"Appears to be identifier/free text ({cardinality} unique) - exclude from release"

        # Default: Top-K for categorical
        return PreprocessingStrategy.TOP_K, \
            f"High cardinality ({cardinality}) categorical - use Top-K binning"

    def calculate_feasibility(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str]
    ) -> Dict:
        """Calculate k-anonymity feasibility metrics."""

        n_records = len(data)
        qi_cardinalities = {}
        qi_product = 1

        for qi in quasi_identifiers:
            if qi in data.columns:
                card = data[qi].nunique()
                qi_cardinalities[qi] = card
                qi_product *= card

        expected_eq = n_records / qi_product if qi_product > 0 else 0

        if expected_eq >= 10:
            feasibility = 'easy'
            max_k = min(int(expected_eq), 20)
        elif expected_eq >= 5:
            feasibility = 'moderate'
            max_k = 5
        elif expected_eq >= 3:
            feasibility = 'hard'
            max_k = 3
        else:
            feasibility = 'infeasible'
            max_k = 0

        return {
            'n_records': n_records,
            'n_qis': len(quasi_identifiers),
            'qi_cardinalities': qi_cardinalities,
            'combination_space': qi_product,
            'expected_eq_size': expected_eq,
            'feasibility': feasibility,
            'max_achievable_k': max_k,
            'exceeds_max_space': qi_product > self.constraints['max_combination_space']
        }

    def preprocess_qis(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        hierarchies: Optional[Dict[str, Dict]] = None,
        auto_preprocess: bool = True,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, List[str], PreprocessingPlan]:
        """
        Preprocess high-cardinality QIs to make k-anonymity feasible.

        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        quasi_identifiers : list
            Original QI column names
        hierarchies : dict, optional
            User-defined preprocessing rules per column:
            {
                'column_name': {
                    'strategy': 'hierarchy' | 'top_k' | 'binning' | 'exclude' | 'reclassify',
                    'mapping': dict or callable,  # for hierarchy
                    'target': str,  # new column name for hierarchy
                    'k': int,  # for top_k
                    'bins': int or list,  # for binning
                    'labels': list,  # for binning labels
                }
            }
        auto_preprocess : bool
            If True, automatically apply recommended strategies
        verbose : bool
            Print preprocessing details

        Returns:
        --------
        tuple : (processed_data, processed_qis, preprocessing_plan)
        """
        hierarchies = hierarchies or {}
        processed_data = data.copy()
        processed_qis = []
        sensitive_vars = []
        excluded_vars = []
        transformations = {}
        warnings = []

        # Calculate initial feasibility
        feasibility_before = self.calculate_feasibility(data, quasi_identifiers)

        if verbose:
            print(f"\n{'='*70}")
            print(f"  QI PREPROCESSING - {self.access_tier.value} ACCESS TIER")
            print(f"{'='*70}")
            print(f"  Constraints: max {self.constraints['max_cardinality_per_qi']} per QI, "
                  f"max {self.constraints['max_qi_count']} QIs")
            print(f"  Initial: {feasibility_before['n_qis']} QIs, "
                  f"{feasibility_before['combination_space']:,} combinations, "
                  f"EQ={feasibility_before['expected_eq_size']:.2f}")
            print(f"  Feasibility: {feasibility_before['feasibility'].upper()}")

        # Analyze all QIs
        analyses = self.analyze_qis(data, quasi_identifiers)

        for qi in quasi_identifiers:
            if qi not in data.columns:
                warnings.append(f"QI '{qi}' not found in data")
                continue

            analysis = analyses.get(qi)

            # Get user-defined or auto strategy
            if qi in hierarchies:
                user_config = hierarchies[qi]
                strategy = PreprocessingStrategy(user_config.get('strategy', 'keep'))
            elif auto_preprocess and analysis and analysis.exceeds_threshold:
                strategy = analysis.recommended_strategy
            else:
                strategy = PreprocessingStrategy.KEEP

            if verbose and strategy != PreprocessingStrategy.KEEP:
                print(f"\n  {qi}: {analysis.cardinality} unique -> {strategy.value}")

            # Apply strategy
            if strategy == PreprocessingStrategy.KEEP:
                processed_qis.append(qi)
                transformations[qi] = {'strategy': 'keep', 'original_cardinality': analysis.cardinality}

            elif strategy == PreprocessingStrategy.HIERARCHY:
                new_col, new_data = self._apply_hierarchy(
                    processed_data, qi, hierarchies.get(qi, {})
                )
                processed_data = new_data
                processed_qis.append(new_col)
                new_card = processed_data[new_col].nunique()
                transformations[qi] = {
                    'strategy': 'hierarchy',
                    'original_cardinality': analysis.cardinality,
                    'new_column': new_col,
                    'new_cardinality': new_card
                }
                if verbose:
                    print(f"    -> {new_col}: {new_card} unique values")

            elif strategy == PreprocessingStrategy.TOP_K:
                k = hierarchies.get(qi, {}).get('k', min(30, self.constraints['max_cardinality_per_qi']))
                new_col = self._apply_top_k(processed_data, qi, k)
                processed_qis.append(new_col)
                new_card = processed_data[new_col].nunique()
                transformations[qi] = {
                    'strategy': 'top_k',
                    'original_cardinality': analysis.cardinality,
                    'k': k,
                    'new_column': new_col,
                    'new_cardinality': new_card
                }
                if verbose:
                    print(f"    -> {new_col}: top {k} + Other = {new_card} values")

            elif strategy == PreprocessingStrategy.BINNING:
                bins = hierarchies.get(qi, {}).get('bins', 10)
                labels = hierarchies.get(qi, {}).get('labels', None)
                new_col = self._apply_binning(processed_data, qi, bins, labels)
                processed_qis.append(new_col)
                new_card = processed_data[new_col].nunique()
                transformations[qi] = {
                    'strategy': 'binning',
                    'original_cardinality': analysis.cardinality,
                    'bins': bins,
                    'new_column': new_col,
                    'new_cardinality': new_card
                }
                if verbose:
                    print(f"    -> {new_col}: {new_card} bins")

            elif strategy == PreprocessingStrategy.RECLASSIFY:
                sensitive_vars.append(qi)
                transformations[qi] = {
                    'strategy': 'reclassify',
                    'original_cardinality': analysis.cardinality,
                    'moved_to': 'sensitive'
                }
                if verbose:
                    print(f"    -> Moved to sensitive attributes")

            elif strategy == PreprocessingStrategy.EXCLUDE:
                excluded_vars.append(qi)
                if qi in processed_data.columns:
                    processed_data = processed_data.drop(columns=[qi])
                transformations[qi] = {
                    'strategy': 'exclude',
                    'original_cardinality': analysis.cardinality
                }
                if verbose:
                    print(f"    -> Excluded from release")

        # Calculate final feasibility
        feasibility_after = self.calculate_feasibility(processed_data, processed_qis)

        if verbose:
            print(f"\n  {'-'*66}")
            print(f"  RESULT: {len(processed_qis)} QIs, "
                  f"{feasibility_after['combination_space']:,} combinations, "
                  f"EQ={feasibility_after['expected_eq_size']:.2f}")
            print(f"  Feasibility: {feasibility_after['feasibility'].upper()}")

            if feasibility_after['feasibility'] == 'infeasible':
                print(f"\n  [!] WARNING: Still infeasible! Consider:")
                print(f"      - Further aggregation of high-cardinality QIs")
                print(f"      - Reducing number of QIs")
                print(f"      - Using higher k value")

        plan = PreprocessingPlan(
            access_tier=self.access_tier,
            original_qis=quasi_identifiers,
            processed_qis=processed_qis,
            sensitive_vars=sensitive_vars,
            excluded_vars=excluded_vars,
            transformations=transformations,
            feasibility_before=feasibility_before,
            feasibility_after=feasibility_after,
            warnings=warnings
        )

        return processed_data, processed_qis, plan

    def _apply_hierarchy(
        self,
        data: pd.DataFrame,
        column: str,
        config: Dict
    ) -> Tuple[str, pd.DataFrame]:
        """Apply hierarchical generalization."""

        mapping = config.get('mapping')
        target = config.get('target', f"{column}_generalized")

        if mapping is None:
            # No mapping provided - use top_k as fallback for categorical
            if pd.api.types.is_numeric_dtype(data[column]):
                data[target] = pd.cut(data[column], bins=10, labels=False)
            else:
                # Fallback: use top_k (keep most frequent values + 'Other')
                # Be aggressive - aim for ~10-15 values to ensure feasibility
                cardinality = data[column].nunique()
                k = min(15, max(5, cardinality // 10))
                top_k_values = data[column].value_counts().head(k).index.tolist()
                data[target] = data[column].apply(
                    lambda x: x if x in top_k_values else 'Other'
                )
        elif callable(mapping):
            data[target] = data[column].apply(mapping)
        elif isinstance(mapping, dict):
            data[target] = data[column].map(mapping).fillna('Other')
        else:
            data[target] = data[column]

        return target, data

    def _apply_top_k(
        self,
        data: pd.DataFrame,
        column: str,
        k: int
    ) -> str:
        """Apply Top-K + Other binning."""

        new_col = f"{column}_topk"
        top_k_values = data[column].value_counts().head(k).index.tolist()
        data[new_col] = data[column].apply(
            lambda x: x if x in top_k_values else 'Other'
        )
        return new_col

    def _apply_binning(
        self,
        data: pd.DataFrame,
        column: str,
        bins: Union[int, List],
        labels: Optional[List] = None
    ) -> str:
        """Apply numeric binning."""

        new_col = f"{column}_binned"

        if pd.api.types.is_numeric_dtype(data[column]):
            if isinstance(bins, int):
                data[new_col] = pd.qcut(
                    data[column], q=bins, labels=labels, duplicates='drop'
                )
            else:
                data[new_col] = pd.cut(data[column], bins=bins, labels=labels)
        else:
            # For non-numeric, use frequency-based binning
            value_counts = data[column].value_counts()
            if isinstance(bins, int):
                # Group into bins by cumulative frequency
                cumsum = value_counts.cumsum() / len(data)
                bin_edges = [i/bins for i in range(bins + 1)]
                bin_map = {}
                for val in value_counts.index:
                    pct = cumsum[val]
                    bin_idx = min(int(pct * bins), bins - 1)
                    bin_map[val] = f"Group_{bin_idx + 1}"
                data[new_col] = data[column].map(bin_map)
            else:
                data[new_col] = data[column]

        return new_col


def preprocess_for_anonymization(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    access_tier: str = 'SCIENTIFIC',
    hierarchies: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str], PreprocessingPlan]:
    """
    Convenience function to preprocess data for anonymization.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : list
        QI column names
    access_tier : str
        'PUBLIC', 'SCIENTIFIC', or 'SECURE'
    hierarchies : dict, optional
        User-defined preprocessing rules
    verbose : bool
        Print progress

    Returns:
    --------
    tuple : (processed_data, processed_qis, plan)

    Example:
    --------
    >>> processed, qis, plan = preprocess_for_anonymization(
    ...     data,
    ...     ['age', 'municipality', 'occupation_code'],
    ...     access_tier='PUBLIC',
    ...     hierarchies={
    ...         'municipality': {
    ...             'strategy': 'hierarchy',
    ...             'mapping': muni_to_region_dict,
    ...             'target': 'region'
    ...         },
    ...         'occupation_code': {
    ...             'strategy': 'top_k',
    ...             'k': 30
    ...         }
    ...     }
    ... )
    """
    handler = QIHandler(access_tier=access_tier)
    return handler.preprocess_qis(
        data, quasi_identifiers,
        hierarchies=hierarchies,
        verbose=verbose
    )
