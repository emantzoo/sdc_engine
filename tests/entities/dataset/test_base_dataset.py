
import pytest
from sdc_engine.entities.dataset.base import BaseDataset

def test_abstract_base_class():
    with pytest.raises(TypeError):
        BaseDataset()