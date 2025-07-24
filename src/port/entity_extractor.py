from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import pandas as pd
import logging
import time

from utils.preprocessing import convert_X_to_list, convert_y_to_list
from utils.typings import TextInput, OutputType



class SingleEntityExtractor(BaseEstimator,ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Taking stats for training and prediction. Prediction is list of times to compute average later on
        self._stats = {
            "fit":{
                "time": 0,
                "samples": 0,
                "words": 0,
            }, 
            "predict_time": [], # List of times to compute average later on
        }
        self.logger = logging.getLogger(__name__)

    
    def fit(self, X: TextInput, y: TextInput=None):
        start_time = time.time()
        X_ = convert_X_to_list(X)
        y_ = convert_y_to_list(y) if y else None
        self.logger.info(f"Fitting {self.__class__.__name__} with {len(X_)} samples")
        self._stats["fit"]["time"] = time.time() - start_time
        self._stats["fit"]["samples"] = len(X_)
        self._stats["fit"]["words"] = sum(len(x.split()) for x in X_)
        return self._fit(X_, y_)
    
    def predict(self, X:TextInput) -> OutputType:
        start_time = time.time()
        X_ = convert_X_to_list(X)
        self.logger.info(f"Predicting {self.__class__.__name__} with {len(X_)} samples")
        self._stats["predict_time"].append(time.time() - start_time)
        return self._predict(X_)
    
    def fit_predict(self, X:TextInput, y:TextInput=None):
        X_ = convert_X_to_list(X)
        y_ = convert_y_to_list(y) if y else None
        self.logger.info(f"Fitting and predicting {self.__class__.__name__} with {len(X_)} samples")
        return self.fit(X_, y_).predict(X_)
    
    @abstractmethod
    def _fit(self, X:TextInput, y:TextInput=None):
        return self

    @abstractmethod
    def _predict(self, X:TextInput):
        return []
    
    @property
    def stats(self):
        return self._stats
    

class MultiEntityExtractor(BaseEstimator,ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractors: dict[str, SingleEntityExtractor] = {}
        self.logger = logging.getLogger(__name__)


    def add_extractor(self, name:str, extractor:SingleEntityExtractor):
        self.extractors[name] = extractor
    
    def remove_extractor(self, name:str):
        del self.extractors[name]
    
    def get_extractor(self, name:str) -> SingleEntityExtractor:
        return self.extractors[name]
    
    def get_all_extractors(self) -> dict[str, SingleEntityExtractor]:
        return self.extractors
    
    def get_all_extractor_names(self) -> list[str]: 
        return list(self.extractors.keys())

    def fit(self, X:TextInput, y:TextInput=None):
        self.logger.info(f"Fitting {self.__class__.__name__} with {len(X)} samples")
        return self._fit(X, y)
    
    def predict(self, X:TextInput) -> OutputType:
        self.logger.info(f"Predicting {self.__class__.__name__} with {len(X)} samples")
        return self._predict(X)
    
    def fit_predict(self, X:TextInput, y:TextInput=None):
        self.logger.info(f"Fitting and predicting {self.__class__.__name__} with {len(X)} samples")
        return self.fit(X).predict(X)
    
    def _fit(self, X:TextInput, y:TextInput=None):
        for name, extractor in self.extractors.items():
            extractor.fit(X, y)
            self.extractors[name] = extractor
        return self
    
    def _predict(self, X:TextInput):
        df = pd.DataFrame()
        for name, extractor in self.extractors.items():
            df[name] = extractor.predict(X)
        return df