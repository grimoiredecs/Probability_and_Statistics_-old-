"""Bayesian and generalised linear regression strategies."""

from sklearn.linear_model import BayesianRidge, TweedieRegressor

from .base import ModelStrategy


class BayesianRidgeStrategy(ModelStrategy):
    def __init__(self, name: str = "Bayesian Ridge", **params):
        super().__init__(name, params)
        self.model = BayesianRidge(**params)


class TweedieStrategy(ModelStrategy):
    def __init__(self, name: str = "Tweedie GLM", power: float = 1.5, alpha: float = 0.01, max_iter: int = 1000, **params):
        super().__init__(name, {"power": power, "alpha": alpha, "max_iter": max_iter, **params})
        self.model = TweedieRegressor(power=power, alpha=alpha, max_iter=max_iter, **params)
