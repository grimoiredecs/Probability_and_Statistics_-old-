import pytest
import os
from src.hardware_benchmarking.pipeline import Pipeline, PipelineContext
from src.hardware_benchmarking.pipeline.filters import (
    ValidationFilter,
    EDAFilter,
    DataCleaningFilter,
    FeatureEngineeringFilter,
    ModelTrainingFilter,
    ModelEvaluationFilter,
    RegistryPromotionFilter,
)


def test_cpu_pipeline_execution(tmp_path):
    pipeline = (
        ValidationFilter()
        | EDAFilter(output_dir=str(tmp_path / "eda"))
        | DataCleaningFilter()
        | FeatureEngineeringFilter(store_dir=str(tmp_path / "feature_store"))
        | ModelTrainingFilter(strategies=["linear_regression", "extra_trees"])
        | ModelEvaluationFilter(min_r2=0.50, max_rmse=1.0)
        | RegistryPromotionFilter(models_dir=str(tmp_path / "models"))
    )

    ctx = PipelineContext(
        domain_key="cpu",
        raw_filepath="data/raw/Intel_CPUs.csv",
        target_col="Processor_Base_Frequency",
    )
    res_ctx = pipeline.execute(ctx)

    assert res_ctx.validation_report["status"] == "PASSED"
    assert res_ctx.cleaned_data is not None
    assert res_ctx.X_train is not None
    assert res_ctx.champion_model_name is not None
    assert res_ctx.champion_metrics["R2_Score"] > 0.50
    assert res_ctx.gate_passed is True
