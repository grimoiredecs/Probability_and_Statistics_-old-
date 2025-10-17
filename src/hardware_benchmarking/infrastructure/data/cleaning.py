"""Domain-aware, non-leaking cleaning for raw hardware specification data."""

import re
import unicodedata
from typing import Any, Optional

import numpy as np
import pandas as pd


class DataCleaner:
    """Normalise raw values without fitting statistics on the full dataset.

    Missing feature values deliberately remain missing here. Imputation is a
    fitted transformation in the feature preprocessor, so cross-validation and
    held-out testing never observe medians from their evaluation rows.
    """

    MISSING_TOKENS = {"", "-", "n/a", "na", "none", "null", "unknown", "not available", "nan"}

    CPU_NUMERIC_COLUMNS = (
        "Processor_Base_Frequency", "Max_Turbo_Frequency", "Lithography",
        "Recommended_Customer_Price", "TDP", "Max_Memory_Size",
        "Max_Memory_Bandwidth", "Cache", "nb_of_Cores", "nb_of_Threads",
        "Max_nb_of_Memory_Channels",
    )
    GPU_NUMERIC_COLUMNS = (
        "Core_Speed", "Boost_Clock", "Max_Power", "Memory", "Memory_Bandwidth",
        "Memory_Bus", "Memory_Speed", "Process", "ROPs", "TMUs",
    )

    def __init__(self) -> None:
        self.last_report: dict[str, Any] = {}

    @classmethod
    def normalize_text(cls, value: Any) -> Optional[str]:
        if pd.isna(value) or value is None:
            return None
        text = unicodedata.normalize("NFKC", str(value)).strip()
        text = re.sub(r"\s+", " ", text)
        return None if text.casefold() in cls.MISSING_TOKENS else text

    @classmethod
    def parse_numeric_spec(cls, value: Any) -> Optional[float]:
        """Parse numeric specifications while preserving unknown values as NA."""
        text = cls.normalize_text(value)
        if text is None:
            return None
        # Thousands separators are common in cache, memory, and price fields.
        compact = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
        match = re.search(r"[-+]?\d+(?:\.\d+)?", compact)
        return float(match.group()) if match else None

    @classmethod
    def parse_frequency_ghz(cls, value: Any) -> Optional[float]:
        text = cls.normalize_text(value)
        numeric = cls.parse_numeric_spec(text)
        if numeric is None:
            return None
        return numeric / 1000.0 if "mhz" in text.casefold() else numeric

    def _clean(self, df: pd.DataFrame, target_col: str, numeric_columns: tuple[str, ...], frequency_columns: tuple[str, ...]) -> pd.DataFrame:
        cleaned = df.copy()
        original_rows = len(cleaned)

        object_columns = cleaned.select_dtypes(include=["object", "string", "category"]).columns
        for column in object_columns:
            cleaned[column] = cleaned[column].map(self.normalize_text)

        for column in numeric_columns:
            if column in cleaned.columns:
                parser = self.parse_frequency_ghz if column in frequency_columns else self.parse_numeric_spec
                cleaned[column] = cleaned[column].map(parser)

        duplicates = int(cleaned.duplicated().sum())
        cleaned = cleaned.drop_duplicates().copy()

        # Hardware resource quantities and targets cannot be non-positive.
        positive_columns = [column for column in numeric_columns if column in cleaned.columns]
        invalid_counts: dict[str, int] = {}
        for column in positive_columns:
            invalid = cleaned[column].notna() & (~np.isfinite(cleaned[column]) | (cleaned[column] <= 0))
            invalid_counts[column] = int(invalid.sum())
            cleaned.loc[invalid, column] = np.nan

        before_target_drop = len(cleaned)
        cleaned = cleaned.dropna(subset=[target_col]).copy()
        self.last_report = {
            "input_rows": original_rows,
            "duplicate_rows_removed": duplicates,
            "invalid_values_converted_to_missing": invalid_counts,
            "rows_without_valid_target_removed": before_target_drop - len(cleaned),
            "output_rows": len(cleaned),
            "remaining_missing_values": cleaned.isna().sum()[cleaned.isna().sum() > 0].to_dict(),
        }
        return cleaned

    def clean_cpu_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = self._clean(df, "Processor_Base_Frequency", self.CPU_NUMERIC_COLUMNS,
                              ("Processor_Base_Frequency", "Max_Turbo_Frequency"))
        if "Launch_Date" in cleaned.columns:
            quarters = cleaned["Launch_Date"].astype(str).str.extract(r"Q([1-4])'(\d{2})")
            years = 2000 + pd.to_numeric(quarters[1], errors="coerce")
            months = (pd.to_numeric(quarters[0], errors="coerce") - 1) * 3 + 1
            cleaned["available_at"] = pd.to_datetime({"year": years, "month": months, "day": 1}, errors="coerce")
            cleaned = cleaned.dropna(subset=["available_at"])
        self.last_report["temporal_rows_removed"] = int(self.last_report["output_rows"] - len(cleaned))
        self.last_report["output_rows"] = len(cleaned)
        return cleaned

    def clean_gpu_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = self._clean(df, "Core_Speed", self.GPU_NUMERIC_COLUMNS, ())
        if "Release_Date" in cleaned.columns:
            cleaned["available_at"] = pd.to_datetime(cleaned["Release_Date"], errors="coerce")
            cleaned = cleaned.dropna(subset=["available_at"])
        self.last_report["temporal_rows_removed"] = int(self.last_report["output_rows"] - len(cleaned))
        self.last_report["output_rows"] = len(cleaned)
        return cleaned
