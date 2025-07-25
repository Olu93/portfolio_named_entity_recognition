from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import pandas as pd
import logging
import time
import numpy as np

from utils.misc import compute_macro_metrics, compute_micro_metrics
from utils.preprocessing import convert_X_to_list, convert_y_to_list
from utils.typings import TextInput, OutputType



class SingleEntityExtractor(BaseEstimator,ABC):
    def __init__(self):
        # super().__init__(*args, **kwargs)
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
        y_ = None if y is None else convert_y_to_list(y)
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
        metrics =self.evaluate(y_, y_hat)
        self._stats["evaluation"]["time"] = time.time() - start_time
        self._stats["evaluation"]["samples"] = len(X_)
        self._stats["evaluation"]["words"] = sum(len(x.split()) for x in X_)
        self._stats["evaluation"]["metrics"] = metrics
        return _self
    
    def predict(self, X:TextInput) -> OutputType:
        X_ = convert_X_to_list(X)
        self.logger.info(f"Predicting {self.__class__.__name__} with {len(X_)} samples")
        return self._predict(X_)
    
    def evaluate(self, y:TextInput, y_hat:OutputType):
        self.logger.info(f"Evaluating {self.__class__.__name__} with {len(y)} samples")
        res = self._evaluate(y, y_hat)
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
    
    # TODO: Need to implement two evaluation procedures
    # Note that names and locations should be evaluated only by matching multi-word entities, i.e., "Joe Biden" is only true if your prediction is "Joe Biden", and not if your prediction is ["Joe", "Biden"]. For organizations, please evaluate using both multi-word prediction (i.e. "Department of Agriculture" is True if the true label is “Department of Agriculture”) and single-word comparison, i.e., ["Department", “of”, "Agriculture"] compared to every single word in the true label.
    def _evaluate(self, y:TextInput, y_hat:OutputType):
        precision_macro, recall_macro, jaccard_macro, f1_macro, accuracy_macro = compute_macro_metrics(y, y_hat)
        precision_micro, recall_micro, jaccard_micro, f1_micro, accuracy_micro = compute_micro_metrics(y, y_hat)

        return {
            "micro": {
                "accuracy": accuracy_micro,
                "precision": precision_micro,
                "recall": recall_micro,
                "jaccard": jaccard_micro,
                "f1": f1_micro,
            },
            "macro": {
                "accuracy": accuracy_macro,
                "precision": precision_macro,
                "recall": recall_macro,
                "jaccard": jaccard_macro,
                "f1": f1_macro,
            }
        }
    
    @property
    def stats(self):
        return self._stats
    

class MultiEntityExtractor(BaseEstimator,ABC):
    def __init__(self):
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

    def fit(self, X:TextInput, Y:pd.DataFrame=None):
        self.logger.info(f"Fitting {self.__class__.__name__} with {len(X)} samples")
        return self._fit(X, Y)
    
    def predict(self, X:TextInput) -> OutputType:
        self.logger.info(f"Predicting {self.__class__.__name__} with {len(X)} samples")
        return self._predict(X)
    
    def fit_predict(self, X:TextInput, y:TextInput=None):
        self.logger.info(f"Fitting and predicting {self.__class__.__name__} with {len(X)} samples")
        return self.fit(X).predict(X)
    
    def _fit(self, X:TextInput, Y:pd.DataFrame=None):
        for name, extractor in self.extractors.items():
            extractor.fit(X, Y[name])
            self.extractors[name] = extractor
        return self
    
    def _predict(self, X:TextInput):
        df = pd.DataFrame()
        for name, extractor in self.extractors.items():
            df[name] = extractor.predict(X)
        return df