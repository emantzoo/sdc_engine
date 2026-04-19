# Empirical Validation Harness

Evaluates whether the rule-engine thresholds in `sdc_engine/sdc/selection/rules.py` produce optimal method selections on real datasets. The harness does NOT auto-update thresholds -- it only reports.

## Quick start

1. Place dataset files in `tests/empirical/data/` (gitignored)
2. Register them in `tests/empirical/datasets.py`
3. Run:

```bash
python -m tests.empirical.run_validation
```

## Adding a dataset

Edit `tests/empirical/datasets.py` and add a `DatasetSpec`:

```python
DatasetSpec(
    name="my_dataset",
    path=DATA_DIR / "my_dataset.csv",
    quasi_identifiers=["age", "gender", "region"],
    sensitive_columns=["income"],
    description="Description of the dataset",
    relevant_thresholds=["T1", "T2", "T4"],
)
```

Set `relevant_thresholds` to the threshold IDs that this dataset is useful for testing. Leave empty to test against all thresholds.

## Thresholds tested

| ID | Name | Current | Rule |
|----|------|---------|------|
| T1 | RC1 dominated cutoff | 0.40 | `risk_concentration_rules` |
| T2 | CAT1 categorical ratio | 0.70 | `categorical_aware_rules` |
| T3 | QR2 suppression gate | 0.25 | `reid_risk_rules` |
| T4 | LOW1 reid_95 gate | 0.10 | `low_risk_rules` |

## Options

```
--thresholds T1 T2    Only test specific thresholds
--datasets adult      Only test specific datasets
--risk-target 0.05    Risk target (default 0.05)
--out path/to/dir     Output directory (default tests/empirical/reports/latest)
```

## Output

- `reports/latest/results.csv` -- one row per (dataset x threshold x value)
- `reports/latest/crossovers.csv` -- where method selection changes
- `reports/latest/report.md` -- human-readable summary

## Interpreting results

A **crossover** means the selected method changes when a threshold value is varied. If the observed crossover point differs from the current threshold by more than 5 percentage points, the report flags it as `CONSIDER`.

A threshold should only be changed if:
- At least 2 datasets show crossover in the same direction
- The shift from current threshold is >5 percentage points
- Utility/ReID outcomes at the proposed new value are better or comparable
