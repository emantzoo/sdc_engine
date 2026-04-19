from sdc_engine.entities.dataset.base import BaseDataset
from dataclasses import dataclass
import pandas as pd
import io, os, tempfile
@dataclass
class LoadDataSet:
    def load_from_csv(self, BaseDataset ,file_obj)-> None:
        BaseDataset.set(pd.read_csv(file_obj, dtype=str, na_values=[], keep_default_na=False))
        
    def load_from_xlsx(self, BaseDataset, file_obj)-> BaseDataset:
        BaseDataset.set(pd.read_excel(file_obj, dtype=str, na_values=[], keep_default_na=False))
        
    def load_from_db(self, BaseDataset, BaseDBAdaptor)-> BaseDataset:
        BaseDataset.set(BaseDBAdaptor.get_data_frame())
        
    