from abc import ABC, abstractmethod
from typing import Any, List,Dict
import pandas as pd


class BaseReidentificationRisc(ABC):
    @abstractmethod
    def compute(self, data:Any)->float:
        pass
    
    @abstractmethod
    def compute_per_entity(self, data:Any, **kwargs )->pd.DataFrame: #TODO refactor to a basic type eg list of tuples
        pass    