from typing import List
from .context import PipelineContext
from .base import BaseFilter


class Pipeline:
    """
    Pipes & Filters Pipeline Orchestrator (Mark Richards Architecture Pattern).
    Sequentially passes PipelineContext through a chain of modular processing Filters.
    """

    def __init__(self, filters: List[BaseFilter] = None):
        self.filters: List[BaseFilter] = filters if filters is not None else []

    def add_filter(self, filter_component: BaseFilter) -> "Pipeline":
        self.filters.append(filter_component)
        return self

    def __or__(self, next_stage) -> "Pipeline":
        if isinstance(next_stage, BaseFilter):
            self.filters.append(next_stage)
        elif isinstance(next_stage, Pipeline):
            self.filters.extend(next_stage.filters)
        return self

    def execute(self, context: PipelineContext) -> PipelineContext:
        print(f"\n>>> EXECUTING PIPELINE CHAIN ({len(self.filters)} Filters) for '{context.domain_key.upper()}' <<<")

        try:
            for idx, filter_component in enumerate(self.filters, 1):
                print(f"\n--- [Filter {idx}/{len(self.filters)}] {filter_component.name} ---")
                context = filter_component.process(context)
            context.mark_completed()
            print(f"\n>>> PIPELINE EXECUTED SUCCESSFULLY in {context.execution_time_sec}s <<<")
            return context
        finally:
            if context.tracker:
                context.tracker.finish(failed=not context.gate_passed)
