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


# USER: populate this list with your datasets.
# Each dataset should span at least one threshold's range of values.
DATASETS: List[DatasetSpec] = [
    # Example:
    # DatasetSpec(
    #     name="adult",
    #     path=DATA_DIR / "adult.csv",
    #     quasi_identifiers=["age", "workclass", "education", "marital-status",
    #                        "occupation", "race", "sex", "native-country"],
    #     sensitive_columns=["income"],
    #     description="UCI Adult -- mixed categorical/continuous",
    #     relevant_thresholds=["T1", "T2", "T4"],
    # ),
]
