"""
Dataset registry for empirical validation.

The user places CSV/Excel files in tests/empirical/data/ (gitignored)
and registers them here with metadata about which thresholds they exercise.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class DatasetSpec:
    name: str
    path: Path
    quasi_identifiers: List[str]
    sensitive_columns: List[str] = field(default_factory=list)
    description: str = ""
    # Which threshold IDs this dataset is useful for exercising
    relevant_thresholds: List[str] = field(default_factory=list)

    def load(self) -> pd.DataFrame:
        if self.path.suffix == ".csv":
            return pd.read_csv(self.path)
        return pd.read_excel(self.path)


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        name="testdata",
        path=DATA_DIR / "testdata.csv",
        quasi_identifiers=["roof", "walls", "water", "electcon",
                           "relat", "sex", "urbrur"],
        sensitive_columns=["income"],
        description="sdcMicro testdata -- categorical-heavy (7 cat QIs), 4580 rows",
        relevant_thresholds=["T1", "T2", "T4"],
    ),
    DatasetSpec(
        name="CASCrefmicrodata",
        path=DATA_DIR / "CASCrefmicrodata.csv",
        quasi_identifiers=["AGI", "FEDTAX", "PTOTVAL", "PEARNVAL", "ERNVAL"],
        sensitive_columns=[],
        description="sdcMicro CASC -- all continuous financial vars, 1080 rows",
        relevant_thresholds=["T1", "T3"],
    ),
    DatasetSpec(
        name="free1",
        path=DATA_DIR / "free1.csv",
        quasi_identifiers=["REGION", "SEX", "AGE", "MARSTAT", "ETNI",
                           "EDUC1", "EDUC2"],
        sensitive_columns=["INCOME"],
        description="sdcMicro free1 -- mixed cat/cont (5 cat + 2 cont QIs), 4000 rows",
        relevant_thresholds=["T1", "T2", "T3", "T4"],
    ),
    # UCI Adult boundary datasets — same CSV, different QI sets targeting
    # specific threshold decision boundaries
    DatasetSpec(
        name="adult_low_reid",
        path=DATA_DIR / "adult.csv",
        quasi_identifiers=["age", "sex", "race"],
        sensitive_columns=["income"],
        description="UCI Adult 3 QIs -- reid_95=0.07, cat=0.67 (LOW1 boundary)",
        relevant_thresholds=["T4"],
    ),
    DatasetSpec(
        name="adult_mid_reid",
        path=DATA_DIR / "adult.csv",
        quasi_identifiers=["age", "sex", "race", "marital_status"],
        sensitive_columns=["income"],
        description="UCI Adult 4 QIs -- reid_95=0.25, cat=0.75 (CAT1/QR2 boundary)",
        relevant_thresholds=["T2", "T3"],
    ),
    DatasetSpec(
        name="adult_high_reid",
        path=DATA_DIR / "adult.csv",
        quasi_identifiers=["age", "sex", "race", "education"],
        sensitive_columns=["income"],
        description="UCI Adult 4 QIs -- reid_95=0.50, cat=0.75 (QR2 boundary)",
        relevant_thresholds=["T3"],
    ),
    # Greek real-estate -- all-categorical boundary datasets
    DatasetSpec(
        name="greek_low_reid",
        path=DATA_DIR / "greek_realestate.csv",
        quasi_identifiers=["\u039d\u03bf\u03bc\u03b1\u03c1\u03c7\u03af\u03b1",
                           "\u039a\u03b1\u03c4\u03b7\u03b3\u03bf\u03c1\u03af\u03b1 \u0391\u03ba\u03b9\u03bd\u03ae\u03c4\u03bf\u03c5",
                           "\u03a0\u03bb\u03ae\u03b8\u03bf\u03c2 \u03a0\u03c1\u03bf\u03c3\u03cc\u03c8\u03b5\u03c9\u03bd"],
        sensitive_columns=["\u03a4\u03af\u03bc\u03b7\u03bc\u03b1 \u0394\u03b9\u03ba\u03b1\u03b9\u03ce\u03bc\u03b1\u03c4\u03bf\u03c2"],
        description="Greek RE 3 QIs -- reid_95=0.125, cat=1.00 (T4 boundary)",
        relevant_thresholds=["T4"],
    ),
    DatasetSpec(
        name="greek_mid_reid",
        path=DATA_DIR / "greek_realestate.csv",
        quasi_identifiers=["\u039d\u03bf\u03bc\u03b1\u03c1\u03c7\u03af\u03b1",
                           "\u039a\u03b1\u03c4\u03b7\u03b3\u03bf\u03c1\u03af\u03b1 \u0391\u03ba\u03b9\u03bd\u03ae\u03c4\u03bf\u03c5",
                           "\u03a0\u03bb\u03ae\u03b8\u03bf\u03c2 \u03a0\u03c1\u03bf\u03c3\u03cc\u03c8\u03b5\u03c9\u03bd",
                           "\u038c\u03c1\u03bf\u03c6\u03bf\u03c2"],
        sensitive_columns=["\u03a4\u03af\u03bc\u03b7\u03bc\u03b1 \u0394\u03b9\u03ba\u03b1\u03b9\u03ce\u03bc\u03b1\u03c4\u03bf\u03c2"],
        description="Greek RE 4 QIs -- reid_95=0.33, cat=1.00 (T2/T3 boundary)",
        relevant_thresholds=["T2", "T3"],
    ),
]
