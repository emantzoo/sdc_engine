"""
Spec 16a -- Generate harness expansion fixture datasets.

Each fixture is designed to trigger a specific pipeline or rule.
Run once to produce CSV files in tests/empirical/data/.

Strategy: Instead of naive random generation, we CONSTRUCT datasets by
explicitly creating QI value combinations and assigning controlled
equivalence-class sizes. This guarantees exact reid_95, high_risk_rate,
feasibility, and other metrics needed to trigger each rule.

Findings (pre-Fix 0): DYN_CAT_Pipeline and CAT2_Mixed_Categorical are
unreachable -- both gated to l_diversity but use NOISE (blocked for
l_diversity). GEO1 was also unreachable due to GENERALIZE missing from
METRIC_ALLOWED_METHODS (config bug, fixed 2026-04-20). Post-Fix 0,
GEO1 fires correctly. DYN_CAT/CAT2 remain unreachable (design question).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from itertools import product as cartesian_product

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
np.random.seed(42)


def g1_ldiv1():
    """G1: LDIV1_Low_Sensitive_Diversity rule.

    Trigger (rules.py line 1110-1227):
    - risk_metric not in ('k_anonymity', 'uniqueness')  [use reid95]
    - sensitive_column_diversity <= 5
    - has_reid = True, has quasi_identifiers
    - NOT infeasible (feasibility != 'infeasible' AND est_supp_k3 <= 0.50)
    - estimated_min_l < 2 (injected via feature_overrides in verify)
    - PRAM allowed for the metric (reid95: yes)

    Must avoid ALL earlier rules in the chain:
    - Pipeline rules (P5 is the risk): needs density>=5 OR n_cont<2 OR n_cat<2
    - RC rules: need N>10000 (no var_priority) OR reid<=0.15
    - MED1: need reid<=0.20 or avoid moderate pattern
    - CAT1/CAT2: only fire for l_diversity metric (we use reid95)

    Strategy: 4 cat + 1 cont. cat_ratio=4/5=0.80 (>=0.70 -> cat-dominant guard
    blocks DYN builder, and P5 requires n_cont>=2 which fails with 1 cont).
    N>10000 avoids var_priority. Large eq -> low reid (<= 0.20).
    """
    n_target = 30000

    cat_vals = {
        'occupation': ['white_collar', 'blue_collar', 'service'],
        'education': ['low', 'medium', 'high'],
        'marital': ['single', 'married', 'divorced'],
        'housing': ['own', 'rent', 'social'],
    }
    cont_vals = {
        'age': list(range(22, 43)),  # 21 values (>20 -> continuous)
    }

    all_cols = list(cat_vals.keys()) + list(cont_vals.keys())
    all_lists = [cat_vals[k] for k in cat_vals] + [cont_vals[k] for k in cont_vals]
    all_combos = list(cartesian_product(*all_lists))
    n_combos = len(all_combos)  # 3*3*3*3*21 = 1701

    # eq = 30000/1701 = 17.6. easy. Low reid.
    # density = 30000/1701 = 17.6 >= 5. P5 won't fire.
    base_size = n_target // n_combos  # 17
    sizes = np.full(n_combos, base_size)
    remainder = n_target - sizes.sum()
    if remainder > 0:
        bonus_idx = np.random.choice(n_combos, remainder, replace=True)
        for idx in bonus_idx:
            sizes[idx] += 1

    rows = []
    for combo, size in zip(all_combos, sizes):
        for _ in range(size):
            rows.append(combo)
    df = pd.DataFrame(rows, columns=all_cols)
    n = len(df)
    # Sensitive column with low diversity (3 values)
    df['disease'] = np.random.choice(['cancer', 'diabetes', 'none'], n)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_DIR / "fixture_g1_ldiv1.csv", index=False)
    print(f"G1: {len(df)} rows, {n_combos} combos, expected_eq={len(df)/n_combos:.1f}, "
          f"density={len(df)/n_combos:.1f}, disease unique={df['disease'].nunique()}")


def g2_geo1():
    """G2: GEO1 pipeline.

    Trigger (pipelines.py line 111-138):
    - len(geo_qis_by_granularity) >= 2
    - has_fine (card > 50) AND has_coarse (card <= 50)
    - reid_95 > 0.15
    - NOT infeasible

    BUT: build_dynamic_pipeline has cat_ratio >= 0.70 guard (line 64) that
    returns False BEFORE reaching GEO1 check (line 111). With all-categorical
    QIs, cat_ratio=1.0 and GEO1 is unreachable!

    Fix: Add a continuous QI to bring cat_ratio < 0.70.
    With city(cat) + region(cat) + income(cont): cat_ratio = 2/3 = 0.67 < 0.70.
    But then DYN_CAT might fire (0.50<0.67<0.70 AND l_diversity).
    Under reid95 metric, DYN_CAT won't fire (needs l_diversity).
    And 0.50<cat_ratio<0.70 enters DYN_CAT check... but metric gate blocks it.
    So GEO1 check (line 111) is reached. Good.

    Combo: 60 * 10 * 22 = 13200. Need N large enough for feasibility.
    N=50000. eq = 50000/13200 = 3.8. hard. NOT infeasible.
    """
    n_target = 50000

    # Fine-grained geo (>50 unique -> 'fine')
    cities = [f"city_{i}" for i in range(60)]
    # Coarse geo (<=50 unique -> 'coarse')
    regions = [f"region_{i}" for i in range(10)]
    # Continuous QI to bring cat_ratio below 0.70
    income_vals = list(range(20, 42))  # 22 values (>20 -> continuous)

    all_cols = ['city', 'region', 'income']
    all_lists = [cities, regions, income_vals]
    all_combos = list(cartesian_product(*all_lists))
    n_combos = len(all_combos)  # 60*10*22 = 13200

    base_size = n_target // n_combos  # ~3
    sizes = np.full(n_combos, base_size)
    remainder = n_target - sizes.sum()
    if remainder > 0:
        bonus_idx = np.random.choice(n_combos, remainder, replace=True)
        for idx in bonus_idx:
            sizes[idx] += 1

    # Some singletons for reid_95 > 0.15
    thin_count = int(n_combos * 0.10)
    thin_idx = np.random.choice(n_combos, thin_count, replace=False)
    surplus = 0
    for idx in thin_idx:
        if sizes[idx] > 1:
            surplus += sizes[idx] - 1
            sizes[idx] = 1
    thick_idx = np.array([i for i in range(n_combos) if i not in thin_idx])
    if surplus > 0 and len(thick_idx) > 0:
        bonus = np.random.choice(thick_idx, surplus, replace=True)
        for idx in bonus:
            sizes[idx] += 1

    rows = []
    for combo, size in zip(all_combos, sizes):
        for _ in range(size):
            rows.append(combo)
    df = pd.DataFrame(rows, columns=all_cols)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_DIR / "fixture_g2_geo1.csv", index=False)
    print(f"G2: {len(df)} rows, {n_combos} combos, expected_eq={len(df)/n_combos:.1f}, "
          f"city unique={df['city'].nunique()}, region unique={df['region'].nunique()}, "
          f"income unique={df['income'].nunique()}")


def g3_dyn_pipeline():
    """G3: DYN pipeline (kANON + LOCSUPR).

    Trigger:
    - reid_95 > 0.20 (adds kANON)
    - cat_ratio < 0.70
    - high_risk_rate > 0.30 AND est_supp_k7 < 0.40 (adds LOCSUPR at k=7)
    - NOT infeasible
    - Pipeline has 2+ methods

    Strategy: 1 cat (4 values) + 2 cont (22 unique each).
    Engineered bimodal eq distribution: 35% records in classes<5, rest in size>=7.
    Outliers injected for has_outliers flag.
    """
    n_target = 10000

    cat_vals = {'category': ['A', 'B', 'C', 'D']}
    cont_vals = {
        'var1': list(range(100, 166, 3)),   # 22 values
        'var2': list(range(200, 266, 3)),   # 22 values
    }

    all_cols = list(cat_vals.keys()) + list(cont_vals.keys())
    all_lists = [cat_vals[k] for k in cat_vals] + [cont_vals[k] for k in cont_vals]
    all_combos = list(cartesian_product(*all_lists))
    n_combos = len(all_combos)  # 4 * 22 * 22 = 1936

    tiny_count = 1550
    large_count = n_combos - tiny_count  # 386

    tiny_sizes = []
    tiny_sizes.extend([1] * 500)
    tiny_sizes.extend([2] * 400)
    tiny_sizes.extend([3] * 350)
    tiny_sizes.extend([4] * 300)
    tiny_records = sum(tiny_sizes)  # 3550

    large_total = n_target - tiny_records  # 6450
    large_base = large_total // large_count
    large_sizes = [large_base] * large_count
    large_remainder = large_total - sum(large_sizes)
    for i in range(large_remainder):
        large_sizes[i] += 1

    sizes = np.array(tiny_sizes + large_sizes)
    perm = np.random.permutation(n_combos)
    shuffled_sizes = sizes[perm]

    rows = []
    for i, combo in enumerate(all_combos):
        for _ in range(shuffled_sizes[i]):
            rows.append(combo)
    df = pd.DataFrame(rows, columns=all_cols)

    # Inject outliers
    n = len(df)
    outlier_count = int(n * 0.05)
    outlier_records = []
    for _ in range(outlier_count):
        cat_val = np.random.choice(cat_vals['category'])
        v1 = np.random.choice([300, 350, 400, 450, 500])
        v2 = np.random.choice(cont_vals['var2'])
        outlier_records.append((cat_val, v1, v2))

    df_outliers = pd.DataFrame(outlier_records, columns=all_cols)
    df = pd.concat([df, df_outliers], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_DIR / "fixture_g3_dyn.csv", index=False)
    print(f"G3: {len(df)} rows, var1 unique={df['var1'].nunique()}, "
          f"var2 unique={df['var2'].nunique()}")


def g4_p4b():
    """G4: P4b pipeline (skewed + sensitive).

    Trigger: skewed_count >= 2, has_sensitive_attributes, n_qis >= 2,
    NOT infeasible, sensitive_column_diversity <= 10.
    """
    n = 10000

    raw_income = np.random.lognormal(10, 1.2, n)
    income_max = np.percentile(raw_income, 99)
    income_bins = np.linspace(raw_income.min(), income_max, 31)
    income = np.digitize(raw_income, income_bins) * 5000

    raw_wealth = np.random.lognormal(11, 1.5, n)
    wealth_max = np.percentile(raw_wealth, 99)
    wealth_bins = np.linspace(raw_wealth.min(), wealth_max, 26)
    wealth = np.digitize(raw_wealth, wealth_bins) * 20000

    df = pd.DataFrame({
        'income': income,
        'wealth': wealth,
        'sex': np.random.choice(['M', 'F'], n),
        'disease': np.random.choice(['diabetes', 'heart', 'cancer',
                                      'none', 'asthma'], n),
    })
    df.to_csv(DATA_DIR / "fixture_g4_p4.csv", index=False)
    print(f"G4: {len(df)} rows, income unique={df['income'].nunique()}, "
          f"wealth unique={df['wealth'].nunique()}, "
          f"income skew={df['income'].skew():.2f}, wealth skew={df['wealth'].skew():.2f}")


def g5_p5():
    """G5: P5 pipeline (sparse, mixed).

    Trigger: n_records >= 200, density < 5, uniqueness > 0.15,
    n_continuous >= 2, n_categorical >= 2, NOT infeasible.
    """
    n_target = 18000

    cat_vals = {
        'district': ['D1', 'D2', 'D3', 'D4'],
        'sector': ['public', 'private', 'ngo'],
    }
    cont_vals = {
        'salary': list(range(20, 42)),   # 22 values
        'experience': list(range(0, 22)),  # 22 values
    }

    all_cols = list(cat_vals.keys()) + list(cont_vals.keys())
    all_lists = [cat_vals[k] for k in cat_vals] + [cont_vals[k] for k in cont_vals]
    all_combos = list(cartesian_product(*all_lists))
    n_combos = len(all_combos)  # 5808

    base_size = n_target // n_combos
    sizes = np.full(n_combos, base_size)
    remainder = n_target - sizes.sum()
    if remainder > 0:
        bonus_idx = np.random.choice(n_combos, remainder, replace=True)
        for idx in bonus_idx:
            sizes[idx] += 1

    thin_count = int(n_combos * 0.12)
    thin_idx = np.random.choice(n_combos, thin_count, replace=False)
    surplus = 0
    for idx in thin_idx:
        if sizes[idx] > 1:
            surplus += sizes[idx] - 1
            sizes[idx] = 1
    thick_idx = np.array([i for i in range(n_combos) if i not in thin_idx])
    if surplus > 0 and len(thick_idx) > 0:
        bonus = np.random.choice(thick_idx, surplus, replace=True)
        for idx in bonus:
            sizes[idx] += 1

    rows = []
    for combo, size in zip(all_combos, sizes):
        for _ in range(size):
            rows.append(combo)
    df = pd.DataFrame(rows, columns=all_cols)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_DIR / "fixture_g5_p5.csv", index=False)
    print(f"G5: {len(df)} rows, {n_combos} combos, density={len(df)/n_combos:.2f}")


def g6_cat1():
    """G6: CAT1 under l_diversity.

    Trigger: risk_metric='l_diversity', 0.15<=reid_95<=0.40, cat_ratio>=0.70,
    no dominant category, NOT infeasible.
    """
    n_target = 20000

    cat_vals = {
        'education': ['primary', 'secondary', 'tertiary', 'vocational'],
        'marital': ['single', 'married', 'divorced', 'widowed'],
        'employment': ['employed', 'unemployed', 'retired', 'student'],
        'housing': ['own', 'rent', 'social'],
    }
    cont_vals = {
        'age': list(range(20, 41)),  # 21 values
    }

    all_cols = list(cat_vals.keys()) + list(cont_vals.keys())
    all_lists = [cat_vals[k] for k in cat_vals] + [cont_vals[k] for k in cont_vals]
    all_combos = list(cartesian_product(*all_lists))
    n_combos = len(all_combos)  # 4032

    base_size = n_target // n_combos
    sizes = np.full(n_combos, base_size)
    remainder = n_target - sizes.sum()
    if remainder > 0:
        bonus_idx = np.random.choice(n_combos, remainder, replace=True)
        for idx in bonus_idx:
            sizes[idx] += 1

    thin_count = int(n_combos * 0.08)
    thin_idx = np.random.choice(n_combos, thin_count, replace=False)
    surplus = 0
    for idx in thin_idx:
        new_size = np.random.choice([1, 2])
        surplus += sizes[idx] - new_size
        sizes[idx] = new_size
    thick_idx = np.array([i for i in range(n_combos) if i not in thin_idx])
    if surplus > 0 and len(thick_idx) > 0:
        bonus = np.random.choice(thick_idx, surplus, replace=True)
        for idx in bonus:
            sizes[idx] += 1

    rows = []
    for combo, size in zip(all_combos, sizes):
        for _ in range(size):
            rows.append(combo)
    df = pd.DataFrame(rows, columns=all_cols)
    n = len(df)
    df['health_status'] = np.random.choice(['good', 'fair', 'poor'], n)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_DIR / "fixture_g6_cat1.csv", index=False)
    print(f"G6: {len(df)} rows, {n_combos} combos, expected_eq={len(df)/n_combos:.1f}")


def g7_low2():
    """G7: LOW2_Continuous_Noise rule.

    Trigger (rules.py line 697-716):
    - reid_95 <= 0.20 (low risk)
    - cat_ratio <= 0.40 (continuous-dominant)
    - n_cont > 0
    - is_very_low (reid_95 <= 0.05) OR (has_outliers AND reid_95 <= 0.10)

    Must also avoid pipeline rules (P5 needs n_cat>=2 AND n_cont>=2 AND density<5)
    and avoid DP1 (which fires when has_outliers AND n_cont>0, at position 13,
    AFTER low_risk_rules at position 12). Wait: checking chain order:
    low_risk_rules is at position 12, distribution_rules at position 13.
    So LOW2 fires BEFORE DP1. If LOW2 conditions are met, it returns first.

    Problem: LOW2 needs reid_95 <= 0.20 AND (is_very_low OR has_outliers+reid<=0.10).
    DP1 needs has_outliers AND n_cont>0 (no reid gate).
    If reid_95 > 0.20: LOW2 doesn't fire, DP1 would fire (at position 13).
    If reid_95 <= 0.10 AND has_outliers: LOW2 fires first (position 12). GOOD.

    So we need: reid_95 <= 0.10, has_outliers, cat_ratio <= 0.40, n_cont > 0.
    AND avoid earlier rules: pipeline_rules=False, structural=False, etc.

    Strategy: 1 cat + 3 cont QIs. cat_ratio=1/4=0.25.
    Large eq for low reid. Outliers in one continuous column.
    Combo with all QIs as constructed values from fixed sets:
    3 * 22 * 22 * 22 = 31944 combos. Need N >> 31944 for low reid.
    That's too many. Use fewer cont unique values... but need >20 for continuous.

    Alternative: 0 cat + 3 cont. cat_ratio=0. n_cont=3.
    combo = 22*22*22 = 10648. N=100000. eq=9.4. moderate.
    reid_95: with eq~9.4, most records in classes of 8-11.
    1/9 = 0.11. 95th percentile ~ 0.11. Close to 0.10.
    Need eq bigger: N=200000. eq=18.8. reid_95 ~ 0.05. very_low!
    But 200K rows is excessive.

    Better: 0 cat + 2 cont. combo=22*22=484. N=10000. eq=20.7. easy.
    reid_95 ~ 1/20 = 0.05. very_low! cat_ratio=0/2=0.0. n_cont=2.
    P5 needs n_cat>=2: fails (n_cat=0). Good.
    has_outliers from injected extreme values in one column.

    But wait: with 0 categorical QIs, there are no 'top_categorical_qis' for
    some rules. Let's keep it simple: 0 cat, 2 cont QIs.
    cat_ratio = 0/2 = 0.0 <= 0.40. n_cont=2>0. very_low=True (reid<=0.05).
    LOW2_Continuous_Noise fires!
    """
    n_target = 10000

    cont_vals = {
        'var1': list(range(100, 144, 2)),   # 22 values
        'var2': list(range(200, 244, 2)),   # 22 values
    }

    all_cols = list(cont_vals.keys())
    all_lists = [cont_vals[k] for k in cont_vals]
    all_combos = list(cartesian_product(*all_lists))
    n_combos = len(all_combos)  # 22*22 = 484

    # eq = 10000/484 = 20.7. easy. reid very low.
    base_size = n_target // n_combos  # ~20
    sizes = np.full(n_combos, base_size)
    remainder = n_target - sizes.sum()
    if remainder > 0:
        bonus_idx = np.random.choice(n_combos, remainder, replace=True)
        for idx in bonus_idx:
            sizes[idx] += 1

    rows = []
    for combo, size in zip(all_combos, sizes):
        for _ in range(size):
            rows.append(combo)
    df = pd.DataFrame(rows, columns=all_cols)

    # Inject outliers into var1 (5% extreme values beyond IQR)
    n = len(df)
    outlier_count = int(n * 0.05)
    outlier_idx = np.random.choice(n, outlier_count, replace=False)
    # Normal range: 100-142. IQR ~ 110-132. Q3+1.5*IQR ~ 132+33=165.
    # Values > 165 are outliers.
    df.loc[outlier_idx, 'var1'] = np.random.choice([180, 200, 220, 250, 300],
                                                    outlier_count)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_DIR / "fixture_g7_low2.csv", index=False)
    print(f"G7: {len(df)} rows, {n_combos} base combos, "
          f"var1 unique={df['var1'].nunique()}, var2 unique={df['var2'].nunique()}")


def g8_floor():
    """G8: Floor regime.

    Requires: expected_eq_size <= 4.
    """
    n = 500
    df = pd.DataFrame({
        'dept': np.random.choice([f"dept_{i}" for i in range(10)], n),
        'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], n),
        'shift': np.random.choice(['morning', 'afternoon', 'evening',
                                    'night', 'rotating', 'flex', 'standard'], n),
        'salary': np.random.randint(20000, 80000, n),
    })
    df.to_csv(DATA_DIR / "fixture_g8_floor.csv", index=False)
    print(f"G8: {len(df)} rows, combo=560, expected_eq<1")


def g9_rc4():
    """G9: RC4_Single_Bottleneck (GENERALIZE → kANON pipeline).

    Trigger (rules.py line 451-487):
    - var_priority exists (from backward elimination)
    - risk_concentration pattern = 'balanced' (not dominated/concentrated/spread_high)
    - n_high == 1 (exactly one QI labeled HIGH, contrib 15-39%)
    - n_other >= 3 (at least 3 non-HIGH QIs)
    - reid_95 > 0.15

    IMPORTANT: var_priority is INJECTED in verify_fixtures.py, not computed
    via backward elimination. The RC4 contribution pattern (1 HIGH 15-39%,
    3+ LOW <3%) is mathematically unreachable via backward elimination on
    synthetic data under the 10K-row limit. See investigation writeup:
    docs/investigations/spec_16_readiness_rc_family_preemption.md

    The dataset is realistic (postcode bottleneck + low-card categoricals)
    but the var_priority injection simulates a regime where backward
    elimination would produce the bottleneck pattern — testing that Fix 0
    unblocked RC4's GENERALIZE pipeline at the rule-selection layer.
    """
    n_target = 3000

    # High-cardinality bottleneck QI — postcode with 40 values
    postcodes = [f"PC{i:03d}" for i in range(40)]
    # Low-cardinality QIs
    sex_vals = ['M', 'F']
    marital_vals = ['single', 'married', 'divorced']
    education_vals = ['primary', 'secondary', 'tertiary']

    all_cols = ['postcode', 'sex', 'marital', 'education']
    all_lists = [postcodes, sex_vals, marital_vals, education_vals]
    all_combos = list(cartesian_product(*all_lists))
    n_combos = len(all_combos)  # 40 * 2 * 3 * 3 = 720

    base_size = n_target // n_combos
    sizes = np.full(n_combos, max(1, base_size))
    remainder = n_target - sizes.sum()
    if remainder > 0:
        bonus_idx = np.random.choice(n_combos, remainder, replace=True)
        for idx in bonus_idx:
            sizes[idx] += 1

    rows = []
    for combo, size in zip(all_combos, sizes):
        for _ in range(size):
            rows.append(combo)
    df = pd.DataFrame(rows, columns=all_cols)
    n = len(df)
    df['disease'] = np.random.choice(['cancer', 'diabetes', 'asthma', 'none'], n)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATA_DIR / "fixture_g9_rc4.csv", index=False)
    print(f"G9: {len(df)} rows, {n_combos} combos, expected_eq={len(df)/n_combos:.1f}, "
          f"postcode unique={df['postcode'].nunique()}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    g1_ldiv1()
    g2_geo1()
    g3_dyn_pipeline()
    g4_p4b()
    g5_p5()
    g6_cat1()
    g7_low2()
    g8_floor()
    g9_rc4()
    print("\nAll fixtures generated.")
