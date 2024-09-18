from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import util
from ds_library.models.model_cv import ModelCV
from ds_library.models.model import Model
from ds_library.models.model_type import ModelType
from ds_library.data_preprocessing.normalizer import Normalizer


class GridSearchCVCustom(ModelCV):
    """Custom GridSearchCV implementation for model hyperparameter tuning.

    Parameters
    ----------
    ModelCV : ABC
        Abstract class
    """
    def __init__(self,
                 base_model: Model,
                 normalizer: Normalizer,
                 param_grid: Dict[str, Any],
                 scoring: str,
                 cv: StratifiedKFold,
                 verbose: int = 0,
                 n_jobs: int = 0) -> None:
        """Initialize the GridSearchCVCustom object.

        Parameters
        ----------
        base_model : Model
            The base model to be optimized.
        normalizer : Normalizer
            The normalizer object to preprocess the data.
        param_grid : Dict[str, Any]
            Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
        scoring : str
            Scoring method to evaluate the predictions on the test set.
        cv : StratifiedKFold
            Cross-validation generator or an iterable, determines the cross-validation splitting strategy.
        verbose : int, optional
            Verbosity level, by default 0.
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 0 (no parallelization).
        """
        self._base_model: Model = base_model
        self.scoring: str = scoring
        self.metrics: pd.Series = None

        self.model = GridSearchCV(
            Pipeline([(self.SCALER, normalizer),
                      (self.BASE_MODEL, base_model)]),
                     param_grid = param_grid,
                     scoring=scoring,
                     cv=cv,
                     verbose=verbose,
                     n_jobs=-n_jobs
        )

    def type(self) -> ModelType:
        """Return the type of the base model."""
        return self._base_model.type()
    
    def alias_model(self) -> str:
        """Return the alias string of the base model."""
        return self._base_model.alias_model()

    def get_best_mean_score(self) -> float:
        """Return the mean score of the best model configuration."""
        return self.model.cv_results_[self.MEAN_TEST_SCORE][self._best_arg_score()]

    def get_best_mean_std_score(self) -> float:
        """Return the standard deviation of the mean score of the best model configuration."""
        return self.model.cv_results_[self.STD_TEST_SCORE][self._best_arg_score()]

    def _best_arg_score(self) -> int:
        """Return the index of the best model configuration based on the rank test score."""
        return np.argmin(self.model.cv_results_[self.RANK_TEST_SCORE])
