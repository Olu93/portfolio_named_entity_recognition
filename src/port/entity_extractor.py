from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import pandas as pd
import logging
import time
import numpy as np

from utils.preprocessing import convert_X_to_list, convert_y_to_list
from utils.typings import TextInput, OutputType



class SingleEntityExtractor(BaseEstimator,ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Taking stats for training and prediction. Prediction is list of times to compute average later on
        self._stats = {
            "training":{
                "time": 0,
                "samples": 0,
                "words": 0,
            }, 
            "inference": {
                "time": 0,
                "samples": 0,
                "words": 0,
            },
            "evaluation": {
                "time": 0,
                "samples": 0,
                "words": 0,
            },
        }
        self.logger = logging.getLogger(__name__)

    
    def fit(self, X: TextInput, y: TextInput=None):
        start_time = time.time()
        X_ = convert_X_to_list(X)
        y_ = convert_y_to_list(y) if y else None
        _self = self._fit(X_, y_)
        self.logger.info(f"Fitting {self.__class__.__name__} with {len(X_)} samples")
        self._stats["training"]["time"] = time.time() - start_time
        self._stats["training"]["samples"] = len(X_)
        self._stats["training"]["words"] = sum(len(x.split()) for x in X_)

        start_time = time.time()
        y_hat = self._predict(X_)
        self._stats["inference"]["time"] = time.time() - start_time
        self._stats["inference"]["samples"] = len(X_)
        self._stats["inference"]["words"] = sum(len(x.split()) for x in X_)
        
        start_time = time.time()
        # self.evaluate(y_, y_hat)
        self._stats["evaluation"]["time"] = time.time() - start_time
        self._stats["evaluation"]["samples"] = len(X_)
        self._stats["evaluation"]["words"] = sum(len(x.split()) for x in X_)

        return _self
    
    def predict(self, X:TextInput) -> OutputType:
        X_ = convert_X_to_list(X)
        self.logger.info(f"Predicting {self.__class__.__name__} with {len(X_)} samples")
        return self._predict(X_)
    
    def evaluate(self, y:TextInput, y_hat:OutputType):
        self.logger.info(f"Evaluating {self.__class__.__name__} with {len(y)} samples")
        res = self._evaluate(y, y_hat)
        self._stats["evaluation"] = res
        self.logger.info(f"Evaluation results: {res}")
        return res

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
    
    def _evaluate(self, y:TextInput, y_hat:OutputType):
        y = np.array(y)
        y_hat = np.array(y_hat)
        len_y = len(y)
        len_y_hat = len(y_hat)
        if len_y != len_y_hat:
            raise ValueError(f"Length of y and y_hat must be the same. Got {len_y} and {len_y_hat}")
        
        return {
            "accuracy": np.all(y == y_hat, axis=1).mean(),
            "precision": np.all(y == y_hat, axis=1).mean(), # TODO: Implement precision
            "recall": np.all(y == y_hat, axis=1).mean(), # TODO: Implement recall
            "f1": np.all(y == y_hat, axis=1).mean(), # TODO: Implement f1
        }
    
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