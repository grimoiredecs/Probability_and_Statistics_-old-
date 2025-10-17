import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.base import clone
from ..base import BaseFilter
from ..context import PipelineContext
from ...domain.models.factory import ModelFactory


class ModelTrainingFilter(BaseFilter):
    """
    Filter 5: Model Training Filter (using Model Strategy Factory).
    """

    def __init__(
        self,
        name: str = "Model Training Filter",
        strategies: List[str] = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        super().__init__(name=name)
        self.strategy_names = strategies or [
            "linear_regression",
            "ridge",
            "lasso",
            "elastic_net",
            "bayesian_ridge",
            "huber",
            "ransac",
            "tweedie",
            "random_forest",
            "gradient_boosting",
            "extra_trees",
        ]
        self.cv_folds = cv_folds
        self.random_state = random_state

    def process(self, context: PipelineContext) -> PipelineContext:
        print(f"[{self.name}] Training {len(self.strategy_names)} Classical ML Strategies using Strategy Factory...")
        kf = TimeSeriesSplit(n_splits=self.cv_folds)
        scoring = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error",
        }
        summary = []

        for strat_name in self.strategy_names:
            try:
                strategy = ModelFactory.create(strat_name)
                # Fit imputation, scaling and encoding within each CV fold.
                cv_pipeline = SklearnPipeline([("preprocessor", clone(context.preprocessor)), ("model", strategy.model)])
                cv_results = cross_validate(cv_pipeline, context.raw_X_train, context.y_train, cv=kf, scoring=scoring)
                r2_scores = cv_results["test_r2"]
                summary_row = {
                    "Model": strategy.name,
                    "CV_R2_Mean": float(np.mean(r2_scores)),
                    "CV_R2_Std": float(np.std(r2_scores, ddof=1)),
                    "CV_R2_95CI": float(1.96 * np.std(r2_scores, ddof=1) / np.sqrt(self.cv_folds)),
                    "CV_RMSE_Mean": float(-np.mean(cv_results["test_rmse"])),
                    "CV_MAE_Mean": float(-np.mean(cv_results["test_mae"])),
                    "CV_MAPE_Percent_Mean": float(-100 * np.mean(cv_results["test_mape"])),
                }

                strategy.fit(context.X_train, context.y_train)
                context.fitted_models[strategy.name] = strategy
                summary.append(summary_row)
            except Exception as exc:
                context.candidate_failures[strat_name] = str(exc)
                print(f"[{self.name}] Skipped '{strat_name}': {exc}")

        if not summary:
            raise RuntimeError("Every configured model candidate failed to train.")

        context.cv_summary = pd.DataFrame(summary).sort_values(
            by=["CV_R2_Mean", "CV_RMSE_Mean"], ascending=[False, True]
        )
        print("\n--- MODEL STRATEGY CV COMPARISON ---")
        print(context.cv_summary.to_string(index=False))
        if context.tracker:
            context.tracker.log_params({"strategies": ",".join(self.strategy_names), "cv_folds": self.cv_folds, "split_strategy": "temporal_expanding_window"})
            cv_metrics = {}
            for _, row in context.cv_summary.iterrows():
                model_key = row.Model.lower().replace(" ", "_")
                cv_metrics.update({
                    f"cv_r2_{model_key}": row["CV_R2_Mean"],
                    f"cv_rmse_{model_key}": row["CV_RMSE_Mean"],
                    f"cv_mae_{model_key}": row["CV_MAE_Mean"],
                    f"cv_mape_percent_{model_key}": row["CV_MAPE_Percent_Mean"],
                })
            context.tracker.log_metrics(cv_metrics)
            context.tracker.log_params({f"failed_{name}": reason for name, reason in context.candidate_failures.items()})

        return context
