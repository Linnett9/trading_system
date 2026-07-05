from core.research.ml.models.registry_dependencies import (
    _scikit_learn_dependencies,
    _tree_dependencies,
)
from core.research.ml.models.registry_factory import build_ml_model
from core.research.ml.models.registry_noop import NoOpMLModel
from core.research.ml.models.registry_protocol import IMLModel
from core.research.ml.models.registry_sklearn import (
    LogisticRegressionMLModel,
    TreeClassifierMLModel,
)
