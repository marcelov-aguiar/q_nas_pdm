import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ds_library.models.evaluator.evaluator import Evaluator


class ClassifierEvaluator(Evaluator):
    """Class that contains operations necessary to evaluate a classification model."""
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1'

    def calculate_accuracy_score(self,
                                 y_test: pd.Series,
                                 predictions: pd.Series) -> float:
        """Method that returns the accuracy score.

        Parameters
        ----------
        y_test : pd.Series
            Series containing the true labels for the test set.
        predictions : pd.Series
            Series containing the predicted labels for the test set.

        Returns
        -------
        float
            Returns the accuracy score
        """
        return accuracy_score(y_test, predictions)

    def calculate_precision_score(self,
                                  y_test: pd.Series,
                                  predictions: pd.Series) -> float:
        """Method that returns the precision score.

        Parameters
        ----------
        y_test : pd.Series
            Series containing the true labels for the test set.
        predictions : pd.Series
            Series containing the predicted labels for the test set.

        Returns
        -------
        float
            Returns the precision score
        """
        return precision_score(y_test, predictions)

    def calculate_recall_score(self,
                               y_test: pd.Series,
                               predictions: pd.Series) -> float:
        """Method that returns the recall score.

        Parameters
        ----------
        y_test : pd.Series
            Series containing the true labels for the test set.
        predictions : pd.Series
            Series containing the predicted labels for the test set.

        Returns
        -------
        float
            Returns the recall score
        """
        return recall_score(y_test, predictions)

    def calculate_f1_score(self,
                           y_test: pd.Series,
                           predictions: pd.Series) -> float:
        """Method that returns the f1 score.

        Parameters
        ----------
        y_test : pd.Series
            Series containing the true labels for the test set.
        predictions : pd.Series
            Series containing the predicted labels for the test set.

        Returns
        -------
        float
            Returns the F1 score
        """
        return f1_score(y_test, predictions)

    def evaluate(self,
                 y_test: pd.Series,
                 predictions: pd.Series,
                 verbose: bool = False) -> pd.Series:
        """Evaluate the classification model by calculating accuracy, precision,
        recall, and F1 score.

        Parameters
        ----------
        y_test : pd.Series
            Series containing the true labels for the test set.
        predictions : pd.Series
            Series containing the predicted labels for the test set.
        verbose : bool, optional
            If True, prints the evaluation metrics, by default False

        Returns
        -------
        pd.Series
            Returns a Series containing the evaluation metrics (accuracy, precision, recall, and F1 score) calculated
        """
        metrics = {
            self.ACCURACY: self.calculate_accuracy_score(y_test, predictions),
            self.PRECISION: self.calculate_precision_score(y_test, predictions),
            self.RECALL: self.calculate_recall_score(y_test, predictions),
            self.F1_SCORE: self.calculate_f1_score(y_test, predictions)
        }
        metrics = pd.Series(metrics)
        if verbose:
            print(metrics)
        return metrics