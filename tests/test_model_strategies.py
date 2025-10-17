import numpy as np

from src.hardware_benchmarking.domain.models.factory import ModelFactory


def test_academic_linear_variants_are_registered_and_predict():
    X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0], [5.0, 3.0], [6.0, 3.0]])
    y = np.array([1.1, 2.0, 2.9, 4.2, 5.0, 6.1])

    for name in ["elastic_net", "bayesian_ridge", "huber", "ransac", "tweedie"]:
        strategy = ModelFactory.create(name)
        strategy.fit(X, y)
        predictions = strategy.predict(X)
        assert predictions.shape == y.shape
        assert np.isfinite(predictions).all()
