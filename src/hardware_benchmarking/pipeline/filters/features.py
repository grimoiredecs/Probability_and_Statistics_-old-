from ..base import BaseFilter
from ..context import PipelineContext
from ...infrastructure.data.feature_store import FeatureStore


class FeatureEngineeringFilter(BaseFilter):
    """
    Filter 4: Feature Store & Feature Engineering Filter.
    """

    def __init__(self, name: str = "Feature Engineering Filter", store_dir: str = "feature_store"):
        super().__init__(name=name)
        self.store = FeatureStore(store_dir=store_dir)

    def process(self, context: PipelineContext) -> PipelineContext:
        print(f"[{self.name}] Engineering domain features for '{context.domain_key.upper()}'...")
        if context.domain_key == "cpu":
            context.engineered_data = self.store.engineer_cpu_features(context.cleaned_data)
            table_name = "cpu_features"
        else:
            context.engineered_data = self.store.engineer_gpu_features(context.cleaned_data)
            table_name = "gpu_features"

        table_path = self.store.save_feature_table(context.engineered_data, table_name)
        print(f"[{self.name}] Saved feature table to '{table_path}'.")

        (
            context.X_train,
            context.X_test,
            context.y_train,
            context.y_test,
            context.feature_names,
            context.raw_X_train,
            context.raw_X_test,
        ) = self.store.prepare_train_test_split(
            context.engineered_data,
            target_col=context.target_col,
            domain_key=context.domain_key,
            test_size=0.2,
            random_state=42,
            split_strategy="temporal",
            time_column="available_at",
        )

        context.preprocessor = self.store.preprocessors[context.domain_key]
        if context.tracker:
            context.tracker.log_artifact(table_path)
        print(f"[{self.name}] Prepared train ({context.X_train.shape[0]} rows) and test ({context.X_test.shape[0]} rows) matrices.")
        return context
