"""Non-linear tree ensemble strategies."""

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

from .base import ModelStrategy


class RandomForestStrategy(ModelStrategy):
    def __init__(self, name: str = "Random Forest", n_estimators: int = 200, random_state: int = 42, **params):
        super().__init__(name, {"n_estimators": n_estimators, "random_state": random_state, **params})
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=1, **params)


class GradientBoostingStrategy(ModelStrategy):
    def __init__(self, name: str = "Gradient Boosting", n_estimators: int = 150, learning_rate: float = 0.1, random_state: int = 42, **params):
        super().__init__(name, {"n_estimators": n_estimators, "learning_rate": learning_rate, "random_state": random_state, **params})
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, **params)


class ExtraTreesStrategy(ModelStrategy):
    def __init__(self, name: str = "Extra Trees", n_estimators: int = 200, random_state: int = 42, **params):
        super().__init__(name, {"n_estimators": n_estimators, "random_state": random_state, **params})
        self.model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=1, **params)
