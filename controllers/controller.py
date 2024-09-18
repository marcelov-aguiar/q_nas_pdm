from typing import List
import pandas as pd
from scripts.data_preprocessing.data_loader.data_loader import DataLoader
from scripts.data_preprocessing.preprocessor import PreProcessor
from scripts.models.grid_search_custom import GridSearchCVCustom
from scripts.graphs.plotter import Plotter


class Controller:
    """Class that orchestrates the execution of a list of models."""

    def __init__(self,
                 dataLoader: DataLoader,
                 preprocessor: PreProcessor,
                 models: List[GridSearchCVCustom],
                 plotter: Plotter):
        """Initialize the Controller

        Parameters
        ----------
        dataLoader : DataLoader
            DataLoader object responsible for loading the dataset.
        preprocessor : PreProcessor
            PreProcessor object responsible for preprocessing the dataset.
        models : List[GridSearchCVCustom]
            List of GridSearchCVCustom models to be executed and evaluated.
        plotter : Plotter
            Plotter object responsible for plotting evaluation metrics.
        """
        self._dataLoader = dataLoader
        self._preProcessor = preprocessor
        self._models = models
        self._plotter = plotter

    def run(self):
        """
        Orchestrates the execution of data loading, preprocessing, model training, evaluation, and plotting.
        """
        # Load dataset
        dataset = self._dataLoader.load_dataset()

        # Preprocessing
        x_train, x_test, y_train, y_test = self._preProcessor.preprocess(dataset)

        # Process Models
        for model in self._models:
            self._processModel(model, x_train, x_test, y_train, y_test)
        
        # Evaluate Models
        self._plot_cv_results()
        self._plot_test_metrics()

    def _processModel(self,
                      model: GridSearchCVCustom,
                      x_train: pd.DataFrame,
                      x_test: pd.DataFrame,
                      y_train: pd.DataFrame,
                      y_test: pd.DataFrame):
        """Trains the provided model, predicts on test data, and evaluates its performance.

        Parameters
        ----------
        model : GridSearchCVCustom
            The model to be trained and evaluated.
        x_train : pd.DataFrame
            Training features.
        x_test : pd.DataFrame
            Test features.
        y_train : pd.DataFrame
            Training target.
        y_test : pd.DataFrame
            Test target.
        """
        # Model Training
        model.fit(x_train, y_train)

        # Model Predict
        predictions = model.predict(x_test)

        # Evaluation
        model.evaluate(y_test, predictions, verbose=True)

    def _plot_cv_results(self):
        """
        Plots the cross-validation results for each model.
        """
        model_names = []
        scores = []
        scores_std = []
        for model in self._models:
            model_names.append(model.alias_model())
            scores.append(model.get_best_mean_score())
            scores_std.append(model.get_best_mean_std_score())

        self._plotter.plot(model_names=model_names,
                           metrics=scores,
                           std_devs=scores_std,
                           title="Binary Classification - Data Train")

    def _plot_test_metrics(self):
        """
        Plots the test set evaluation metrics for each model.
        """
        model_names = []
        scores = []
        for model in self._models:
            model_names.append(model.alias_model())
            scores.append(model.get_metrics()[model.scoring])

        self._plotter.plot(model_names=model_names,
                           metrics=scores,
                           title="Binary Classification - Data Test")
