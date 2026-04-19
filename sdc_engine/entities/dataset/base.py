from abc import ABC, abstractmethod
from typing import Any, List

class BaseDataset(ABC):
    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def set(self,data: Any):
        pass

    @abstractmethod
    def get_columns(self)->List[str] |None:
        pass

    @abstractmethod
    def get_risk(self):
        pass
    
    @abstractmethod
    def get_active_columns(self)->List[str] |None:
        pass
    
    @abstractmethod
    def set_active_columns(self, activeCols:List[str])-> None:
        pass
    
    @abstractmethod
    def get_data(self)->Any:
        pass
    
    @abstractmethod
    def no_of_rows(self)->int:
        pass
    
    @abstractmethod
    def no_of_cols(self)->int:        
        pass
    @abstractmethod
    def no_of_active_cols(self)->int:
        pass