"""Explicit registration of the production model catalogue."""

from .ensembles import ExtraTreesStrategy, GradientBoostingStrategy, RandomForestStrategy
from .factory import ModelFactory
from .linear import ElasticNetStrategy, LassoStrategy, LinearRegressionStrategy, RidgeStrategy
from .probabilistic import BayesianRidgeStrategy, TweedieStrategy
from .robust import HuberStrategy, RANSACStrategy


def register_default_strategies() -> None:
    strategies = {
        "linear_regression": LinearRegressionStrategy,
        "ridge": RidgeStrategy,
        "lasso": LassoStrategy,
        "elastic_net": ElasticNetStrategy,
        "bayesian_ridge": BayesianRidgeStrategy,
        "huber": HuberStrategy,
        "ransac": RANSACStrategy,
        "tweedie": TweedieStrategy,
        "random_forest": RandomForestStrategy,
        "gradient_boosting": GradientBoostingStrategy,
        "extra_trees": ExtraTreesStrategy,
    }
    for name, strategy in strategies.items():
        ModelFactory.register(name, strategy)
