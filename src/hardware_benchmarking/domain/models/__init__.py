"""Classical regression strategies, grouped by statistical family."""

from .base import ModelStrategy
from .ensembles import ExtraTreesStrategy, GradientBoostingStrategy, RandomForestStrategy
from .factory import ModelFactory
from .linear import ElasticNetStrategy, LassoStrategy, LinearRegressionStrategy, RidgeStrategy
from .probabilistic import BayesianRidgeStrategy, TweedieStrategy
from .registry import register_default_strategies
from .robust import HuberStrategy, RANSACStrategy

register_default_strategies()

__all__ = [
    "ModelStrategy", "ModelFactory", "LinearRegressionStrategy", "RidgeStrategy", "LassoStrategy",
    "ElasticNetStrategy", "BayesianRidgeStrategy", "TweedieStrategy", "HuberStrategy", "RANSACStrategy",
    "RandomForestStrategy", "GradientBoostingStrategy", "ExtraTreesStrategy", "register_default_strategies",
]
