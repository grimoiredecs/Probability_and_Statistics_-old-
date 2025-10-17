"""Factory used by the training filter to create configured model strategies."""

from typing import Dict, List, Type

from .base import ModelStrategy


class ModelFactory:
    _registry: Dict[str, Type[ModelStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[ModelStrategy]) -> None:
        cls._registry[name] = strategy_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> ModelStrategy:
        if name not in cls._registry:
            raise KeyError(f"Strategy '{name}' is not registered in ModelFactory.")
        return cls._registry[name](**kwargs)

    @classmethod
    def get_registered_names(cls) -> List[str]:
        return list(cls._registry)
