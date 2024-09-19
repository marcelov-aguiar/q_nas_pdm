from enum import Enum
import util
from ds_library.models.evaluator.classifier_evaluator import ClassifierEvaluator
from ds_library.models.evaluator.regression_evaluator import RegressionEvaluator

class ModelType(Enum):
    """Enumerator that represents the model types and their respective evaluators."""
    REGRESSION = RegressionEvaluator()
    CLASSIFICATION = ClassifierEvaluator()
