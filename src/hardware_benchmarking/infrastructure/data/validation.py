import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


class DataValidator:
    """
    Phase 1: Data Validation & Quality Gate Module.
    Ensures raw dataset integrity, checks schema contracts, and detects missing value anomalies.
    """

    MANDATORY_CPU_COLS = [
        "Product_Collection",
        "Vertical_Segment",
        "nb_of_Cores",
        "Processor_Base_Frequency",
        "TDP",
        "Lithography",
    ]

    MANDATORY_GPU_COLS = [
        "Manufacturer",
        "Name",
        "Core_Speed",
        "Memory",
        "Max_Power",
    ]

    def validate_cpu_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates Intel_CPUs schema contract and quality thresholds.
        """
        report: Dict[str, Any] = {"dataset": "Intel_CPUs", "status": "PASSED", "warnings": []}

        if df.empty:
            report["status"] = "FAILED"
            report["error"] = "CPU dataset is empty."
            return report

        # Schema contract check
        missing_cols = [c for c in self.MANDATORY_CPU_COLS if c not in df.columns]
        if missing_cols:
            report["status"] = "FAILED"
            report["error"] = f"Missing mandatory schema columns: {missing_cols}"
            return report

        # Null checks
        null_counts = df[self.MANDATORY_CPU_COLS].isnull().sum().to_dict()
        report["null_counts"] = null_counts

        # Warning threshold check (e.g. target Processor_Base_Frequency missingness > 5%)
        target_null_pct = df["Processor_Base_Frequency"].isnull().mean()
        if target_null_pct > 0.05:
            report["warnings"].append(f"Processor_Base_Frequency missing percentage is high: {target_null_pct:.2%}")

        report["record_count"] = len(df)
        return report

    def validate_gpu_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates All_GPUs schema contract and quality thresholds.
        """
        report: Dict[str, Any] = {"dataset": "All_GPUs", "status": "PASSED", "warnings": []}

        if df.empty:
            report["status"] = "FAILED"
            report["error"] = "GPU dataset is empty."
            return report

        missing_cols = [c for c in self.MANDATORY_GPU_COLS if c not in df.columns]
        if missing_cols:
            report["status"] = "FAILED"
            report["error"] = f"Missing mandatory schema columns: {missing_cols}"
            return report

        null_counts = df[self.MANDATORY_GPU_COLS].isnull().sum().to_dict()
        report["null_counts"] = null_counts
        report["record_count"] = len(df)

        return report
