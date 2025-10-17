"""Outlier-resistant regression strategies."""

from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor

from .base import ModelStrategy


class HuberStrategy(ModelStrategy):
    def __init__(self, name: str = "Huber Regression", epsilon: float = 1.35, max_iter: int = 1000, **params):
        super().__init__(name, {"epsilon": epsilon, "max_iter": max_iter, **params})
        self.model = HuberRegressor(epsilon=epsilon, max_iter=max_iter, **params)


class RANSACStrategy(ModelStrategy):
    def __init__(self, name: str = "RANSAC Regression", max_trials: int = 50, random_state: int = 42, **params):
        super().__init__(name, {"max_trials": max_trials, "random_state": random_state, **params})
        self.model = RANSACRegressor(estimator=LinearRegression(), max_trials=max_trials, random_state=random_state, **params)
