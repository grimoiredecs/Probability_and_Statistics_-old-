from ..base import BaseFilter
from ..context import PipelineContext
from ...infrastructure.data.cleaning import DataCleaner


class DataCleaningFilter(BaseFilter):
    """
    Filter 3: Data Cleaning & Contextual Imputation Filter.
    """

    def __init__(self, name: str = "Data Cleaning Filter"):
        super().__init__(name=name)
        self.cleaner = DataCleaner()

    def process(self, context: PipelineContext) -> PipelineContext:
        print(f"[{self.name}] Cleaning specification strings and applying contextual imputation...")
        if context.domain_key == "cpu":
            context.cleaned_data = self.cleaner.clean_cpu_dataset(context.raw_data)
        else:
            context.cleaned_data = self.cleaner.clean_gpu_dataset(context.raw_data)

        context.cleaning_report = self.cleaner.last_report
        if context.tracker:
            context.tracker.log_params({"cleaning_report": context.cleaning_report})
        print(f"[{self.name}] Cleaned dataset shape: {context.cleaned_data.shape[0]} rows, {context.cleaned_data.shape[1]} columns.")
        return context
