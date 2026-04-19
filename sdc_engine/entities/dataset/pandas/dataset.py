import pandas as pd
from typing import Optional, List
from sdc_engine.entities.dataset.base import BaseDataset
from dataclasses import dataclass


@dataclass
class PdDataset(BaseDataset):
    activeCols: Optional[List[str]] = None
    data: Optional[pd.DataFrame] = None
    distCountsCols: Optional[pd.DataFrame] = None
    
    activeCombinations: Optional[List[str]] = None

    def get(self) -> Optional[pd.DataFrame]:
        return self.data

    def set(self, data: pd.DataFrame) -> None:
        self.data = data

    def get_columns(self) -> List[str]:
        return list(self.data.columns)

    def get_risk(self) -> pd.Series:
        return self.data['risk']

    def get_active_columns(self) -> List[str]:
        activCols = self.get_columns() if self.activeCols is None else self.activeCols
        return activCols
    
    def set_active_columns(self, activeCols:List[str])-> None:
        self.activeCols =activeCols
        

    def get_data(self) -> pd.DataFrame:
        return self.data[self.activeCols]
    
    
    def no_of_rows(self)->int:
        return self.data.shape[0]
    
    
    def no_of_cols(self)->int:
        return self.data.shape[1]
    
    def no_of_active_cols(self)->int:
        return len(self.get_active_columns())
        
