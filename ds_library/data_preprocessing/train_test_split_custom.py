from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


class TrainTestSplitCustom():
    def __init__(self,
                 features: List[str],
                 target: str,
                 test_size: float = 0.3,
                 stratify: bool = False,
                 random_state: int = 42):
        """Splits the data into training and testing sets

        Parameters
        ----------
        features : List[str]
            Names of the input features
        target : str
            Name of the target
        test_size : float, optional
            Size of the test set to be split, by default 0.3
        stratify : bool, optional
            Performs stratified sampling if True to maintain
            class proportions in train/test splits, by default False
        random_state : int, optional
            Controls the randomness of data splitting.
            Set to an int for reproducible results, by default 42
        """
        self._features = features
        self._target = target
        self._test_size = test_size
        self._stratify = stratify
        self._random_state = random_state

    def data_split(self, dataframe: pd.DataFrame) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs the train-test split

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            List with the training and testing data in
            the following sequence: X_train, X_test, y_train, y_test
        """
        X, y = self.separate_feature_and_label(dataframe)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = self._test_size,
                                                            stratify = y if self._stratify else None,
                                                            random_state = self._random_state)


        return X_train, X_test, y_train, y_test

    def separate_feature_and_label(self, dataframe: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
        """To separate the input features from the target

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Data of the features and the target separated in the following sequence: X, y
        """
        if self._target in self._features:
            self._features.remove(self._target)

        X = dataframe[self._features].copy()
        y = dataframe[self._target].copy()
        return X, y
