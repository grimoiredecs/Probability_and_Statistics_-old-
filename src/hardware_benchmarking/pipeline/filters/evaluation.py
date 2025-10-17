from ..base import BaseFilter
from ..context import PipelineContext
from ...infrastructure.registry import ModelRegistry


class ModelEvaluationFilter(BaseFilter):
    """
    Filter 6: Model Testing & Gate Verification Filter.
    """

    def __init__(self, name: str = "Model Evaluation Filter", min_r2: float = 0.70, max_rmse: float = 0.50, max_mape_percent: float | None = None):
        super().__init__(name=name)
        self.registry = ModelRegistry()
        self.min_r2 = min_r2
        self.max_rmse = max_rmse
        self.max_mape_percent = max_mape_percent

    def process(self, context: PipelineContext) -> PipelineContext:
        print(f"[{self.name}] Evaluating candidate strategies on out-of-sample test set...")

        best_model_name = context.cv_summary.iloc[0]["Model"]
        best_strategy = context.fitted_models[best_model_name]

        test_metrics = self.registry.test_model(best_strategy, context.X_test, context.y_test)
        context.test_metrics[best_model_name] = test_metrics

        gate_passed, reasons = self.registry.verify_quality_gate(
            test_metrics, min_r2=self.min_r2, max_rmse=self.max_rmse, max_mape_percent=self.max_mape_percent
        )

        context.gate_passed = gate_passed
        context.reasons = reasons
        context.champion_model_name = best_model_name
        context.champion_model = best_strategy.model
        context.champion_metrics = test_metrics
        if context.tracker:
            context.tracker.log_metrics({f"test_{key.lower()}": value for key, value in test_metrics.items()})

        print(f"[{self.name}] Best Strategy: '{best_model_name}' | Test R²: {test_metrics['R2_Score']:.4f} | Test RMSE: {test_metrics['RMSE']:.4f}")
        print(f"[{self.name}] Quality Gate Verification: {'PASSED' if gate_passed else 'FAILED'}")

        return context
