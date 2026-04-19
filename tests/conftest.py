import pytest
from sdc_engine.entities.dataset.pandas.dataset import PdDataset
from sdc_engine.interactors.load_dataset import LoadDataSet

@pytest.fixture
def dataset():
    base_dataset = PdDataset()
    load_data = LoadDataSet()
    path = "tests/data/datadeclaration_remuneration_only_A.csv"
    load_data.load_from_csv(base_dataset, path)
    return base_dataset
