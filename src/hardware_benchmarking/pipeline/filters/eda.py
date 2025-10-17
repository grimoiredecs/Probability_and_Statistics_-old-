from ..base import BaseFilter
from ..context import PipelineContext
from ...infrastructure.data.eda import EDAEngine


class EDAFilter(BaseFilter):
    """
    Filter 2: Exploratory Data Analysis (EDA) Filter.
    """

    def __init__(self, name: str = "EDA Filter", output_dir: str = "outputs/eda"):
        super().__init__(name=name)
        self.eda_engine = EDAEngine(output_dir=output_dir)

    def process(self, context: PipelineContext) -> PipelineContext:
        print(f"[{self.name}] Running EDA on raw dataset...")
        if context.domain_key == "cpu":
            self.eda_engine.run_cpu_eda(context.raw_data)
        else:
            self.eda_engine.run_gpu_eda(context.raw_data)

        print(f"[{self.name}] EDA visuals exported to '{self.eda_engine.output_dir}'.")
        return context
