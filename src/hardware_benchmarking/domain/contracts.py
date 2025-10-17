"""Stable contracts separating static hardware identity from observations."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class HardwareEntity:
    hardware_id: str
    domain: str
    manufacturer: str | None


@dataclass(frozen=True)
class BenchmarkObservation:
    hardware_id: str
    available_at: datetime
    target_name: str
    target_value: float
    source_id: str
    label_version: str = "v1"
