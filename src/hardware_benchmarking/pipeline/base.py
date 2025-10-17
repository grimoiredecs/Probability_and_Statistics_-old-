from abc import ABC, abstractmethod
from .context import PipelineContext


class BaseFilter(ABC):
    """
    Abstract Filter Component (Mark Richards' Pipes & Filters Architecture Pattern).
    Every processing stage implements a discrete, modular filter taking PipelineContext and returning updated context.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Processes context payload and passes updated state to the next Pipe.
        """
        pass

    def __or__(self, next_filter: "BaseFilter") -> "Pipeline":
        from .orchestrator import Pipeline
        return Pipeline([self, next_filter])
