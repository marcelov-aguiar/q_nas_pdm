from enum import Enum
import util
from ds_library.models.evaluator.classifier_evaluator import ClassifierEvaluator

class ModelType(Enum):
    """Enumerator that represents the model types and their respective evaluators."""
    REGRESSION = None
    CLASSIFICATION = ClassifierEvaluator()
