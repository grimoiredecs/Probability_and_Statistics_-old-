import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


@dataclass
class PipelineContext:
    """
    Pipes & Filters Architecture State Payload (Mark Richards Architecture Pattern).
    Transmits data, features, fitted models, evaluation metrics, and metadata through the pipeline.
    """

    domain_key: str  # 'cpu' or 'gpu'
    raw_filepath: str
    target_col: str

    # Pipeline Payload State
    raw_data: Optional[pd.DataFrame] = None
    cleaned_data: Optional[pd.DataFrame] = None
    engineered_data: Optional[pd.DataFrame] = None

    # Processed Feature Matrices
    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    feature_names: List[str] = field(default_factory=list)
    preprocessor: Optional[Any] = None
    raw_X_train: Optional[pd.DataFrame] = None
    raw_X_test: Optional[pd.DataFrame] = None

    # Model & Evaluation State
    fitted_models: Dict[str, Any] = field(default_factory=dict)
    candidate_failures: Dict[str, str] = field(default_factory=dict)
    cv_summary: Optional[pd.DataFrame] = None
    test_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    champion_model_name: Optional[str] = None
    champion_model: Optional[Any] = None
    champion_metrics: Dict[str, float] = field(default_factory=dict)

    # Status & Diagnostic Metadata
    validation_report: Dict[str, Any] = field(default_factory=dict)
    cleaning_report: Dict[str, Any] = field(default_factory=dict)
    gate_passed: bool = False
    reasons: List[str] = field(default_factory=list)
    execution_time_sec: float = 0.0
    start_timestamp: float = field(default_factory=time.time)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    artifact_dir: Optional[str] = None
    report_path: Optional[str] = None
    tracker: Optional[Any] = None

    def mark_completed(self):
        self.execution_time_sec = round(time.time() - self.start_timestamp, 3)
