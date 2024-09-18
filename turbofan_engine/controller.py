from typing import List
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import util
from ds_library.data_preprocessing.data_loader.data_loader import DataLoader
from ds_library.data_preprocessing.data_loader.data_loader_train_test import DataLoaderTrainTest
from ds_library.data_preprocessing.preprocessor import PreProcessor
from ds_library.data_preprocessing.preprocessor_train_test import EspecificPreprossessorTrainTest
from ds_library.models.grid_search_custom import GridSearchCVCustom
from ds_library.graphs.plotter import Plotter
from ds_library.graphs.matplotter import MatPlotter
from ds_library.data_preprocessing.normalizer_standard_scaler import NormStandardScaler
from ds_library.models.regression.random_forest_regression_model import RandomForestRegressionModel
from ds_library.models.model_cv import ModelCV
from ds_library.models.evaluator.regression_evaluator import RegressionEvaluator
from ds_library.data_preprocessing.data_loader.data_loader_from_txt import DataLoaderFromTXT
from turbofan_engine.data_preprocessing.calc_rul_train import CalcRULTrain
from turbofan_engine.data_preprocessing.calc_rul_test import CalcRULTest
from config.settings import Settings
import turbofan_engine.constants as const

class Controller:
    """Class that orchestrates the execution of a list of models."""

    def __init__(self,
                 dataLoader: DataLoaderTrainTest,
                 preprocessor: EspecificPreprossessorTrainTest,
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

# %%
#### CONSTANTS ####
FEATURES_NAME = const.FEATURES_NAME

# Features to remove from the dataset
FEATURES_TO_REMOVE = ["TEL"]

# Features to be normalized
FEATURES_TO_NORM = ["NDEP", "RENDA", "VBEM", "NPARC", "VPARC", "IDADE", "RESMS", "ENTRADA"]


# Target variable name
TARGET = "target"

settings = Settings()

# Create DataLoader to load the dataset from URL
dataLoader = DataLoaderTrainTest(train_path=const.RAW_PATH_FD001_TRAIN,
                                 test_path=const.RAW_PATH_FD001_TEST,
                                 features_name=FEATURES_NAME)


pipeline_train = Pipeline([(const.CalcRULTrain, CalcRULTrain())])

pipeline_test = Pipeline([(const.CalcRULTest, CalcRULTest(
    DataLoaderFromTXT(source_path=const.RAW_PATH_FD001_RUL,
                      features_name=const.DEFAULT_MAX_NAME).load_dataset()
                      ))
        ])

# Create PreProcessor for data preprocessing
preProcessor = EspecificPreprossessorTrainTest(transformers_train=pipeline_train,
                                               transformers_test=pipeline_test,
                                               features_to_remove=FEATURES_TO_REMOVE,
                                               target=TARGET)

# Create StratifiedKFold cross-validator for model evaluation
stratified_cv = KFold(n_splits=5, shuffle=True, random_state=42)


# Create MatPlotter for plotting evaluation metrics
plotter = MatPlotter()

models = [
    GridSearchCVCustom(
        normalizer = NormStandardScaler(columns_to_norm=FEATURES_TO_NORM),
        base_model = RandomForestRegressionModel(),
        param_grid={
            f'{ModelCV.BASE_MODEL}__n_estimators': [50, 100, 150],
            f'{ModelCV.BASE_MODEL}__max_depth': [None, 10, 20, 30],
            f'{ModelCV.BASE_MODEL}__criterion': ['gini', 'entropy']
        },
        scoring=RegressionEvaluator.RMSE,
        cv=stratified_cv,
        verbose=1,
        n_jobs=-1
    )
]

# Instantiate the Controller to orchestrate the execution of models
controller = Controller(dataLoader, preProcessor, models, plotter)

# Run the Controller to execute data loading, preprocessing, model training, and evaluation
controller.run()