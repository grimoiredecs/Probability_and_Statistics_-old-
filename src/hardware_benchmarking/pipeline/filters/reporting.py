"""Final presentation filter for a completed pipeline run."""

from ..base import BaseFilter
from ..context import PipelineContext
from ...infrastructure.reporting import RunReportGenerator


class ReportExportFilter(BaseFilter):
    """Exports a readable Markdown report and visual assets for evaluators."""

    def __init__(self, name: str = "Report Export Filter", reports_dir: str = "outputs/reports"):
        super().__init__(name=name)
        self.generator = RunReportGenerator(reports_dir)

    def process(self, context: PipelineContext) -> PipelineContext:
        context.report_path = str(self.generator.generate(context))
        if context.tracker:
            context.tracker.log_artifact(context.report_path)
        print(f"[{self.name}] Reader report exported to '{context.report_path}'.")
        return context
