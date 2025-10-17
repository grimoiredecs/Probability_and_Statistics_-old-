import os
import json
import time
import hashlib
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Leakage-safe group median imputer for tabular hardware specifications.

    It is fitted only on a training fold, then falls back to that fold's global
    median for unseen manufacturers or segments.
    """

    def __init__(self, group_column: str, numeric_columns: list[str]):
        self.group_column = group_column
        self.numeric_columns = numeric_columns

    def fit(self, X: pd.DataFrame, y: Any = None):
        frame = X.copy()
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.global_medians_ = frame[self.numeric_columns].median().to_dict()
        self.group_medians_ = (
            frame.groupby(self.group_column, dropna=False)[self.numeric_columns].median().to_dict()
            if self.group_column in frame.columns else {}
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        for column in self.numeric_columns:
            if self.group_medians_ and self.group_column in frame.columns:
                group_values = frame[self.group_column].map(self.group_medians_.get(column, {}))
                frame[column] = frame[column].fillna(group_values)
            frame[column] = frame[column].fillna(self.global_medians_[column])
        return frame

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)


class DynamicCategoryHasher(BaseEstimator, TransformerMixin):
    """Fixed-width encoding for open-ended entities, including unseen values."""

    def __init__(self, n_features: int = 32):
        self.n_features = n_features

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.hasher_ = FeatureHasher(n_features=self.n_features, input_type="string", alternate_sign=False)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        tokens = [[f"{column}={value}" for column, value in row.items() if pd.notna(value)] for _, row in X.iterrows()]
        return self.hasher_.transform(tokens).toarray()

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"hashed_entity_{index}" for index in range(self.n_features)], dtype=object)


class FeatureStore:
    """
    Phase 4: Feature Store & Feature Engineering Module.
    Engineers hardware domain features, persists versioned feature tables in feature_store/,
    and prepares train/test matrices adhering to strict featurization ordering.
    """

    CPU_FEATURE_COLS = [
        "Lithography",
        "Recommended_Customer_Price",
        "nb_of_Cores",
        "nb_of_Threads",
        "Cache",
        "TDP",
        "Max_Memory_Size",
        "Max_Memory_Bandwidth",
        "Max_nb_of_Memory_Channels",
        "Cache_per_Core",
        "TDP_per_Core",
        "TDP_per_Thread",
        "Price_per_Core",
        "Bandwidth_per_Core",
        "Cores_per_Thread_Ratio",
        "Product_Collection",
        "Vertical_Segment",
    ]

    GPU_FEATURE_COLS = [
        "Process",
        "Max_Power",
        "Memory",
        "Memory_Bandwidth",
        "Memory_Bus",
        "Memory_Speed",
        "ROPs",
        "TMUs",
        "Bandwidth_per_Watt",
        "ROPs_to_TMUs_Ratio",
        "Manufacturer",
        "Notebook_GPU",
        "Name",
    ]

    def __init__(self, store_dir: str = "feature_store"):
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.preprocessors: Dict[str, Any] = {}
        self.feature_names: Dict[str, List[str]] = {}

    def engineer_cpu_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives CPU architecture domain features.
        """
        feat_df = df.copy()

        cores = np.where(feat_df["nb_of_Cores"] <= 0, 1, feat_df["nb_of_Cores"])
        threads = np.where(feat_df["nb_of_Threads"] <= 0, cores, feat_df["nb_of_Threads"])

        feat_df["Cache_per_Core"] = feat_df["Cache"] / cores
        feat_df["TDP_per_Core"] = feat_df["TDP"] / cores
        feat_df["TDP_per_Thread"] = feat_df["TDP"] / threads
        if "Recommended_Customer_Price" in feat_df.columns:
            feat_df["Price_per_Core"] = feat_df["Recommended_Customer_Price"] / cores
        if "Max_Memory_Bandwidth" in feat_df.columns:
            feat_df["Bandwidth_per_Core"] = feat_df["Max_Memory_Bandwidth"] / cores

        feat_df["Cores_per_Thread_Ratio"] = cores / threads

        # Replace infinite values if division by zero occurred
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        return feat_df

    def engineer_gpu_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives GPU architecture domain features.
        """
        feat_df = df.copy()

        power = np.where(feat_df["Max_Power"] <= 0, 1, feat_df["Max_Power"])
        tmus = np.where(feat_df["TMUs"] <= 0, 1, feat_df["TMUs"])

        feat_df["Bandwidth_per_Watt"] = feat_df["Memory_Bandwidth"] / power
        feat_df["ROPs_to_TMUs_Ratio"] = feat_df["ROPs"] / tmus
        # Never derive a feature from Core_Speed: it is the GPU prediction target.

        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        return feat_df

    def save_feature_table(self, df: pd.DataFrame, table_name: str) -> str:
        """
        Persists feature table to local feature store with metadata.
        """
        version = time.strftime("%Y%m%dT%H%M%S")
        parquet_path = os.path.join(self.store_dir, f"{table_name}_{version}.parquet")
        df.to_parquet(parquet_path, index=False)

        meta_path = os.path.join(self.store_dir, "features_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                try:
                    meta = json.load(f)
                except Exception:
                    meta = {}

        meta[table_name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "filepath": parquet_path,
            "format": "parquet",
            "sha256": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
            "version": version,
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return parquet_path

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        target_col: str,
        domain_key: str,
        test_size: float = 0.2,
        random_state: int = 42,
        split_strategy: str = "temporal",
        time_column: str = "available_at",
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, List[str]]:
        """
        Splits first, then fits group-aware and fallback imputers only on train rows.
        """
        target_feature_list = self.CPU_FEATURE_COLS if domain_key == "cpu" else self.GPU_FEATURE_COLS
        available_cols = [c for c in target_feature_list if c in df.columns]

        X = df[available_cols].copy()
        y = df[target_col].copy()

        dynamic_cols = [column for column in (["Product_Collection"] if domain_key == "cpu" else ["Name"]) if column in X.columns]
        cat_cols = [column for column in X.select_dtypes(include=["object", "category"]).columns if column not in dynamic_cols]
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if split_strategy == "temporal":
            if time_column not in df.columns or df[time_column].isna().any():
                raise ValueError("Temporal split requires a non-null available_at column.")
            ordered = df.assign(**{time_column: pd.to_datetime(df[time_column])}).sort_values(time_column, kind="stable")
            cut = int(len(ordered) * (1 - test_size))
            train_indices, test_indices = ordered.index[:cut], ordered.index[cut:]
            X_train_raw, X_test_raw = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        else:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        from sklearn.pipeline import Pipeline

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        column_transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols),
                ("dynamic", DynamicCategoryHasher(), dynamic_cols),
            ],
            remainder="drop",
        )

        group_col = "Vertical_Segment" if domain_key == "cpu" else "Manufacturer"
        preprocessor = Pipeline(
            steps=[
                ("group_median_imputer", GroupMedianImputer(group_col, num_cols)),
                ("columns", column_transformer),
            ]
        )
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_test_proc = preprocessor.transform(X_test_raw)
        feature_names = list(preprocessor.get_feature_names_out())
        self.preprocessors[domain_key] = preprocessor
        self.feature_names[domain_key] = feature_names

        return X_train_proc, X_test_proc, y_train, y_test, feature_names, X_train_raw, X_test_raw
