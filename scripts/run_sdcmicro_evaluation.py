"""Run the SDC engine on sdcMicro-bundled datasets and write a markdown report.

Output: docs/sdcmicro_evaluation.md

This is the document a reviewer opens to see concrete results on datasets they
know well.

Usage:
    python scripts/run_sdcmicro_evaluation.py
"""
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.parity.fixtures import load_sdcmicro_dataset
from sdc_engine.sdc.protection_engine import (
    run_rules_engine_protection, build_data_features,
)
from sdc_engine.entities.dataset.pandas.dataset import PdDataset
from sdc_engine.interactors.sdc_protection import SDCProtection

logging.basicConfig(level=logging.WARNING)

EVALUATIONS = [
    {
        "dataset": "testdata",
        "qis": ["roof", "walls", "water", "electcon", "relat", "sex"],
        "sensitive": ["income"],
        "description": (
            "Tanzania household survey -- small, categorical-heavy, typical "
            "sdcMicro tutorial dataset."
        ),
    },
    {
        "dataset": "CASCrefmicrodata",
        "qis": ["AFNLWGT", "AGI", "FEDTAX", "PTOTVAL", "ERNVAL"],
        "sensitive": ["TAXINC"],
        "description": (
            "US Census CASC reference microdata -- medium size, all-continuous."
        ),
    },
    {
        "dataset": "francdat",
        "qis": ["Key1", "Key2", "Key3", "Key4"],
        "sensitive": ["Num1"],
        "description": (
            "Franconi synthetic -- standard SDC benchmark, all-categorical QIs."
        ),
    },
    {
        "dataset": "free1",
        "qis": ["SEX", "MARSTAT", "EDUC1", "AGE"],
        "sensitive": ["INCOME"],
        "description": (
            "Free1 survey -- 4000 records, mixed categorical/continuous QIs."
        ),
    },
]


def run_one(spec: dict) -> dict:
    df = load_sdcmicro_dataset(spec["dataset"])
    qis = [c for c in spec["qis"] if c in df.columns]
    sens = [c for c in spec["sensitive"] if c in df.columns]
    use_cols = qis + sens
    df = df[use_cols].dropna().reset_index(drop=True)

    dataset = PdDataset(data=df.copy())
    dataset.set_active_columns(list(df.columns))
    protector = SDCProtection(dataset=dataset)

    features = build_data_features(df, qis)
    features['_access_tier'] = 'SCIENTIFIC'
    features['_reid_target_raw'] = 0.05
    features['_utility_floor'] = 0.80

    try:
        result, log_entries = run_rules_engine_protection(
            input_data=df, quasi_identifiers=qis,
            data_features=features, access_tier='SCIENTIFIC',
            reid_target=0.05, utility_floor=0.80,
            apply_method_fn=protector.apply_method,
            sensitive_columns=sens, risk_target_raw=0.05,
        )
    except Exception as e:
        return {
            "dataset": spec["dataset"],
            "description": spec["description"],
            "n_records": len(df),
            "n_qis": len(qis),
            "qis": qis,
            "sensitive": sens,
            "method_selected": "ERROR",
            "rule_applied": str(e)[:80],
            "reid_before": 0,
            "reid_after": 0,
            "utility": 0,
            "target_met": False,
        }

    if result is None:
        return {
            "dataset": spec["dataset"],
            "description": spec["description"],
            "n_records": len(df),
            "n_qis": len(qis),
            "qis": qis,
            "sensitive": sens,
            "method_selected": "NONE (all methods failed)",
            "rule_applied": "N/A",
            "reid_before": 0,
            "reid_after": 0,
            "utility": 0,
            "target_met": False,
        }

    meta = result.metadata or {}
    reid_before = (result.reid_before or {}).get('reid_95', 0)
    reid_after = (result.reid_after or {}).get('reid_95', 0)

    # Extract rule from log entries (format: "Rule: RULE_NAME -> METHOD")
    rule_applied = '?'
    for entry in log_entries:
        if entry.startswith('Rule:'):
            rule_applied = entry.split('Rule:')[1].split('\u2192')[0].strip()
            break

    return {
        "dataset": spec["dataset"],
        "description": spec["description"],
        "n_records": len(df),
        "n_qis": len(qis),
        "qis": qis,
        "sensitive": sens,
        "method_selected": getattr(result, 'method', meta.get('method', '?')),
        "rule_applied": rule_applied,
        "reid_before": reid_before,
        "reid_after": reid_after,
        "utility": result.utility_score or 0,
        "target_met": bool(result.success),
    }


def write_report(results: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# SDC Engine Evaluation on sdcMicro Datasets\n\n")
        f.write(f"_Generated {datetime.now().strftime('%Y-%m-%d')}_\n\n")
        f.write("_To regenerate: `python scripts/run_sdcmicro_evaluation.py`_\n\n")
        f.write(
            "This document shows how the SDC engine handles datasets bundled "
            "with the [sdcMicro](https://CRAN.R-project.org/package=sdcMicro) "
            "R package (Templ, Kowarik, Meindl, JSS 2015). "
            "It serves as both a sanity check and a worked-example reference "
            "for reviewers familiar with sdcMicro's conventions.\n\n"
            "All runs use **Scientific Use** tier: target reid_95 <= 5%, "
            "utility_floor >= 80%.\n\n"
        )
        for r in results:
            f.write(f"## {r['dataset']}\n\n")
            f.write(f"{r['description']}\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Records | {r['n_records']:,} |\n")
            f.write(f"| QIs ({r['n_qis']}) | {', '.join(r['qis'])} |\n")
            if r['sensitive']:
                f.write(f"| Sensitive | {', '.join(r['sensitive'])} |\n")
            f.write(f"| Method selected | `{r['method_selected']}` |\n")
            f.write(f"| Rule applied | `{r['rule_applied']}` |\n")
            f.write(f"| ReID95 before | {r['reid_before']:.2%} |\n")
            f.write(f"| ReID95 after | {r['reid_after']:.2%} |\n")
            f.write(f"| Utility | {r['utility']:.1%} |\n")
            f.write(f"| Target met | {'Yes' if r['target_met'] else 'No'} |\n")
            f.write(f"\n")


def main():
    print("Running SDC engine evaluation on sdcMicro datasets...")
    results = []
    for spec in EVALUATIONS:
        print(f"  {spec['dataset']}...", end=" ", flush=True)
        r = run_one(spec)
        print(f"{r['method_selected']} -> reid {r['reid_before']:.2%} -> {r['reid_after']:.2%}")
        results.append(r)

    out = Path(__file__).parent.parent / "docs" / "sdcmicro_evaluation.md"
    write_report(results, out)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
