from ..base import BaseFilter
from ..context import PipelineContext
from ...infrastructure.registry import ModelRegistry


class RegistryPromotionFilter(BaseFilter):
    """
    Filter 7: Model Registry & Artifact Deployment Filter.
    """

    def __init__(self, name: str = "Registry Promotion Filter", models_dir: str = "models"):
        super().__init__(name=name)
        self.registry = ModelRegistry(models_dir=models_dir)

    def process(self, context: PipelineContext) -> PipelineContext:
        if not context.gate_passed:
            print(f"[{self.name}] Quality Gate failed ({context.reasons}). Skipping model registry promotion.")
            return context

        print(f"[{self.name}] Promoting Champion '{context.champion_model_name}' to Production Registry...")
        promoted = self.registry.register_and_promote_champion(
            domain_key=context.domain_key,
            model_name=context.champion_model_name,
            model=context.champion_model,
            preprocessor=context.preprocessor,
            feature_names=context.feature_names,
            metrics=context.champion_metrics,
        )

        status_text = "PROMOTED NEW CHAMPION MODEL" if promoted else "RETAINED EXISTING CHAMPION MODEL"
        if context.tracker and promoted:
            context.tracker.log_artifact(f"models/{context.domain_key}_champion.joblib")
            context.tracker.log_artifact(f"models/{context.domain_key}_preprocessor.joblib")
        print(f"[{self.name}] Registry Result: {status_text}")
        return context
