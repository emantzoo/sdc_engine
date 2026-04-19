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
]
