
import pytest
from sdc_engine.interactors.load_dataset import LoadDataSet
from sdc_engine.entities.dataset.pandas.dataset import PdDataset
import pandas as pd



def test_load_from_csv():
    base_dataset = PdDataset()
    load_data = LoadDataSet()
    path = "tests/data/datadeclaration_remuneration_only_A.csv"
    load_data.load_from_csv(base_dataset, path)
    assert isinstance(base_dataset.get_data(), pd.DataFrame)
    assert base_dataset.get_columns()[1] =="denomination_sociale"

def test_load_from_xlsx():
    base_dataset = PdDataset()
    load_data = LoadDataSet()
    path = "tests/data/declaration_remuneration_only_A.xlsx"
    load_data.load_from_xlsx(base_dataset, path)
    assert isinstance(base_dataset.get_data(), pd.DataFrame)
    assert base_dataset.get_columns()[1] =="denomination_sociale"

@pytest.mark.skip(reason="Not implemented")
def test_load_from_db():
    assert isinstance(result, BaseDataset)
