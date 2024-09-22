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
from ds_library.models.regression.lightgbm_regression_model import LightGBMRegressionModel
from ds_library.models.model_cv import ModelCV
from ds_library.models.evaluator.regression_evaluator import RegressionEvaluator
import ds_library.constants.constants_names as const_names
from ds_library.data_preprocessing.data_loader.data_loader_from_txt import DataLoaderFromTXT
from turbofan_engine.data_preprocessing.rul_calculator.calc_rul_train import CalcRULTrain
from turbofan_engine.data_preprocessing.rul_calculator.calc_rul_test import CalcRULTest
from turbofan_engine.data_preprocessing.rul_calculator.rul_config import RULConfig
from turbofan_engine.data_preprocessing.rul_calculator.rul_merge import DataFrameMerger
from turbofan_engine.data_preprocessing.rul_calculator.rul_calculator import DefaultRULCalculator
from turbofan_engine.data_preprocessing.rul_calculator.rul_filter import RULFilter, RULFilterTwo
from config.settings import Settings
import turbofan_engine.constants as const_turbofan

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
                           title="Data Train")

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
                           title="Data Test")

def main():
    # %%
    #### CONSTANTS ####
    FEATURES_NAME = const_turbofan.FEATURES_NAME

    # Features to remove from the dataset
    FEATURES_TO_REMOVE = [const_turbofan.FEATURE_UNIT_NUMBER]

    # Features to be normalized
    FEATURES_TO_NORM = const_turbofan.FEATURES_TO_NORMALIZER

    # Target variable name
    TARGET = const_turbofan.TARGET

    settings = Settings()

    # Create DataLoader to load the dataset from URL
    dataLoader = DataLoaderTrainTest(train_path=const_turbofan.RAW_PATH_FD001_TRAIN,
                                     test_path=const_turbofan.RAW_PATH_FD001_TEST,
                                     features_name=FEATURES_NAME)

    # Create PreProcessor for data preprocessing
    preProcessor = EspecificPreprossessorTrainTest(
        transformers_train=Pipeline([(const_turbofan.CalcRULTrain, CalcRULTrain()),
                                     (const_turbofan.RULFilter, RULFilter())]),
        transformers_test=Pipeline([
            (const_turbofan.CalcRULTest, 
             CalcRULTest(rul_calculator=DefaultRULCalculator(),
                         merger=DataFrameMerger(),
                         df_rul=DataLoaderFromTXT(
                             path_dataset_txt=const_turbofan.RAW_PATH_FD001_RUL,
                             features_name=[const_turbofan.DEFAULT_MAX_NAME]).load_dataset(),
                         config=RULConfig(
                             feature_unit_number=const_turbofan.FEATURE_UNIT_NUMBER, 
                             feature_time=const_turbofan.FEATURE_TIME, 
                             target=const_turbofan.TARGET, 
                             default_max_name=const_turbofan.DEFAULT_MAX_NAME, 
                             total_rul=const_turbofan.TOTAL_RUL)
             )
            ),
            (const_turbofan.RULFilter, RULFilterTwo())
        ]),
        features_to_remove=FEATURES_TO_REMOVE,
        target=TARGET)

    models = [
        GridSearchCVCustom(
            normalizer = NormStandardScaler(columns_to_norm=FEATURES_TO_NORM),
            base_model = LightGBMRegressionModel(),
            param_grid={
                f'{ModelCV.BASE_MODEL}__n_estimators': [50],
                f'{ModelCV.BASE_MODEL}__max_depth': [-1],
                f'{ModelCV.BASE_MODEL}__num_leaves': ['31']
            },
            scoring=RegressionEvaluator.R2,
            cv=KFold(n_splits=5),
            verbose=1,
            n_jobs=-1
        ),
        # GridSearchCVCustom(
        #     normalizer = NormStandardScaler(columns_to_norm=FEATURES_TO_NORM),
        #     base_model = RandomForestRegressionModel(),
        #     param_grid={
        #         f'{ModelCV.BASE_MODEL}__n_estimators': [50], # 100, 150],
        #         f'{ModelCV.BASE_MODEL}__max_depth': [None], #10],  20, 30],
        #         f'{ModelCV.BASE_MODEL}__criterion': ['absolute_error'] # , 'squared_error']
        #     },
        #     scoring=RegressionEvaluator.R2,
        #     cv=KFold(n_splits=5),
        #     verbose=1,
        #     n_jobs=-1
        # )
    ]

    # Create MatPlotter for plotting evaluation metrics
    plotter = MatPlotter()

    # Instantiate the Controller to orchestrate the execution of models
    controller = Controller(dataLoader, preProcessor, models, plotter)

    # Run the Controller to execute data loading, preprocessing, model training, and evaluation
    controller.run()

if __name__ == "__main__":
    main()