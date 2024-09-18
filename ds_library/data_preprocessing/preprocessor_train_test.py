from typing import List, Tuple
import pandas as pd
import util
from sklearn.pipeline import Pipeline
from ds_library.data_preprocessing.preprocessor import PreProcessor
from ds_library.data_preprocessing.train_test_split_custom import TrainTestSplitCustom
from ds_library.data_preprocessing.data_loader.data_loader_train_test import DataLoaderTrainTest
from ds_library.data_preprocessing.base_transform import BaseTransform


class EspecificPreprossessorTrainTest(PreProcessor):

    def __init__(self,
                 transformers_train: List[BaseTransform],
                 transformers_test: List[BaseTransform],
                 features_to_remove: List[str],
                 target: str):
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
        self._transformers_train = transformers_train
        self._transformers_test = transformers_test

    def preprocess(self, data_loader: DataLoaderTrainTest) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """_summary_

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple containing X_train, X_test, y_train, y_test.
        """
        df_train = data_loader.df_train
        df_test = data_loader.df_test

        for transformer in self._transformers_train:
            df_train = transformer.transform(df_train)
        
        for transformer in self._transformers_test:
            df_test = transformer.transform(df_test)


        X_train, y_train = \
            TrainTestSplitCustom(list(df_train.columns),
                                 self._target).separate_feature_and_label(df_train)
        
        X_test, y_test = \
            TrainTestSplitCustom(list(df_test.columns),
                                 self._target).separate_feature_and_label(df_test)

        return X_train, X_test, y_train, y_test
