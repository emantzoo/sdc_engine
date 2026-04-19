# SDC Engine Evaluation on sdcMicro Datasets

_Generated 2026-04-20_

_To regenerate: `python scripts/run_sdcmicro_evaluation.py`_

This document shows how the SDC engine handles datasets bundled with the [sdcMicro](https://CRAN.R-project.org/package=sdcMicro) R package (Templ, Kowarik, Meindl, JSS 2015). It serves as both a sanity check and a worked-example reference for reviewers familiar with sdcMicro's conventions.

All runs use **Scientific Use** tier: target reid_95 <= 5%, utility_floor >= 80%.

## testdata

Tanzania household survey -- small, categorical-heavy, typical sdcMicro tutorial dataset.

| Metric | Value |
|--------|-------|
| Records | 4,580 |
| QIs (6) | roof, walls, water, electcon, relat, sex |
| Sensitive | income |
| Method selected | `LOCSUPR` |
| Rule applied | `HR3_High_Uniqueness_QIs` |
| ReID95 before | 33.33% |
| ReID95 after | 5.00% |
| Utility | 100.0% |
| Target met | Yes |

## CASCrefmicrodata

US Census CASC reference microdata -- medium size, all-continuous.

| Metric | Value |
|--------|-------|
| Records | 1,080 |
| QIs (5) | AFNLWGT, AGI, FEDTAX, PTOTVAL, ERNVAL |
| Sensitive | TAXINC |
| Method selected | `NOISE` |
| Rule applied | `DP1_Outliers` |
| ReID95 before | 100.00% |
| ReID95 after | 100.00% |
| Utility | 99.3% |
| Target met | Yes |

## francdat

Franconi synthetic -- standard SDC benchmark, all-categorical QIs.

| Metric | Value |
|--------|-------|
| Records | 8 |
| QIs (4) | Key1, Key2, Key3, Key4 |
| Sensitive | Num1 |
| Method selected | `LOCSUPR` |
| Rule applied | `HR6_Very_Small_Dataset` |
| ReID95 before | 100.00% |
| ReID95 after | 50.00% |
| Utility | 100.0% |
| Target met | Yes |

## free1

Free1 survey -- 4000 records, mixed categorical/continuous QIs.

| Metric | Value |
|--------|-------|
| Records | 4,000 |
| QIs (4) | SEX, MARSTAT, EDUC1, AGE |
| Sensitive | INCOME |
| Method selected | `kANON` |
| Rule applied | `HR1_Extreme_Uniqueness` |
| ReID95 before | 100.00% |
| ReID95 after | 12.50% |
| Utility | 100.0% |
| Target met | Yes |

