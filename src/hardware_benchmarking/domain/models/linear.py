"""OLS and regularised linear regression strategies."""

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .base import ModelStrategy


class LinearRegressionStrategy(ModelStrategy):
    def __init__(self, name: str = "Linear Regression", **params):
        super().__init__(name, params)
        self.model = LinearRegression(**params)


class RidgeStrategy(ModelStrategy):
    def __init__(self, name: str = "Ridge Regression", alpha: float = 1.0, **params):
        super().__init__(name, {"alpha": alpha, **params})
        self.model = Ridge(alpha=alpha, **params)


class LassoStrategy(ModelStrategy):
    def __init__(self, name: str = "Lasso Regression", alpha: float = 0.01, max_iter: int = 5000, **params):
        super().__init__(name, {"alpha": alpha, "max_iter": max_iter, **params})
        self.model = Lasso(alpha=alpha, max_iter=max_iter, **params)


class ElasticNetStrategy(ModelStrategy):
    def __init__(self, name: str = "Elastic Net", alpha: float = 0.01, l1_ratio: float = 0.5, max_iter: int = 10000, **params):
        super().__init__(name, {"alpha": alpha, "l1_ratio": l1_ratio, "max_iter": max_iter, **params})
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, **params)
