import pandas as pd
from ..base import BaseFilter
from ..context import PipelineContext
from ...infrastructure.data.validation import DataValidator


class ValidationFilter(BaseFilter):
    """
    Filter 1: Data Validation & Quality Gate Filter.
    Reads raw dataset from raw_filepath and validates schema contract and quality thresholds.
    """

    def __init__(self, name: str = "Data Validation Filter"):
        super().__init__(name=name)
        self.validator = DataValidator()

    def process(self, context: PipelineContext) -> PipelineContext:
        print(f"[{self.name}] Reading raw dataset from '{context.raw_filepath}'...")
        context.raw_data = pd.read_csv(context.raw_filepath)

        if context.domain_key == "cpu":
            context.validation_report = self.validator.validate_cpu_data(context.raw_data)
        else:
            context.validation_report = self.validator.validate_gpu_data(context.raw_data)

        print(f"[{self.name}] Validation Status: {context.validation_report['status']} | Records: {context.validation_report.get('record_count')}")

        if context.validation_report["status"] == "FAILED":
            raise ValueError(f"Validation Filter Failed: {context.validation_report.get('error')}")

        return context
