from typing import Dict
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from ds_library.models.model import Model
from ds_library.models.model import ModelType
from ds_library.models.evaluator.evaluator import Evaluator

class LightGBMRegressionModel(Model):
    def __init__(self, **params: Dict):
        self.model = LGBMRegressor(**params)
        self._alias_model = "LightGBM"

    def type(self) -> Evaluator:
        return ModelType.REGRESSION
