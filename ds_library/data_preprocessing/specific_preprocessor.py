from typing import List, Tuple
import pandas as pd
from preprocessor import PreProcessor
from train_test_split_custom import TrainTestSplitCustom


class EspecificPreprossessor(PreProcessor):

    def __init__(self,
                 features_to_remove: List[str],
                 target: str,
                 columns_one_hot: List[str],
                 seed: int = 42,
                 test_percentage: float = 0.30):
        """_summary_

        Parameters
        ----------
        features_to_remove : List[str]
            List of feature names to be removed from the dataset.
        target : str
            Name of the target column.
        columns_one_hot : List[str]
            List of column names for one-hot encoding.
        seed : int, optional
            Seed for random number generator, by default 42.
        test_percentage : float, optional
            Percentage of data to be allocated for testing, by default 0.30.
        """
        self._features_to_remove = features_to_remove
        self._target = target
        self._columns_one_hot = columns_one_hot
        self._SEED = seed
        self._TEST_PERCENTAGE = test_percentage

    def preprocess(self, dataframe: pd.DataFrame) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """_summary_

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple containing X_train, X_test, y_train, y_test.
        """

        dataframe = dataframe.drop(columns=self._features_to_remove)

        dataframe = pd.get_dummies(dataframe, columns=self._columns_one_hot, dtype=int)

        X_train, X_test, y_train, y_test = \
            TrainTestSplitCustom(list(dataframe.columns),
                                 self._target).data_split(dataframe)

        return X_train, X_test, y_train, y_test
