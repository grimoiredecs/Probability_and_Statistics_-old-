import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List


class EDAEngine:
    """
    Phase 2: Automated Exploratory Data Analysis (EDA) Module.
    Generates summary statistics, correlation structures, and exports EDA visuals to outputs/eda/.
    """

    def __init__(self, output_dir: str = "outputs/eda"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def run_cpu_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes automated EDA on CPU benchmark data.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats_summary = df[num_cols].describe().T.to_dict() if num_cols else {}

        corr_matrix = None
        if "Processor_Base_Frequency" in num_cols:
            corr_matrix = df[num_cols].corr()["Processor_Base_Frequency"].sort_values(ascending=False).to_dict()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if "Processor_Base_Frequency" in df.columns:
            sns.histplot(df["Processor_Base_Frequency"].dropna(), kde=False, ax=axes[0], color="#2b5c8f")
            axes[0].set_title("CPU Base Frequency Spec Distribution")

        if "TDP" in df.columns:
            sns.histplot(df["TDP"].dropna(), kde=False, ax=axes[1], color="#e74c3c")
            axes[1].set_title("CPU TDP Thermal Budget Spec Distribution")

        plt.savefig(os.path.join(self.output_dir, "cpu_eda_distributions.png"), dpi=300, bbox_inches="tight")
        plt.close()

        if len(num_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title("CPU Numeric Feature Correlation Matrix")
            plt.savefig(os.path.join(self.output_dir, "cpu_eda_correlation.png"), dpi=300, bbox_inches="tight")
            plt.close()

        return {"stats": stats_summary, "target_correlations": corr_matrix}

    def run_gpu_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes automated EDA on GPU benchmark data.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats_summary = df[num_cols].describe().T.to_dict() if num_cols else {}

        corr_matrix = None
        if "Core_Speed" in num_cols:
            corr_matrix = df[num_cols].corr()["Core_Speed"].sort_values(ascending=False).to_dict()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if "Core_Speed" in df.columns:
            sns.histplot(df["Core_Speed"].dropna(), kde=False, ax=axes[0], color="#27ae60")
            axes[0].set_title("GPU Core Speed Distribution")

        if "Max_Power" in df.columns:
            sns.histplot(df["Max_Power"].dropna(), kde=False, ax=axes[1], color="#8e44ad")
            axes[1].set_title("GPU Max Power Consumption")

        plt.savefig(os.path.join(self.output_dir, "gpu_eda_distributions.png"), dpi=300, bbox_inches="tight")
        plt.close()

        return {"stats": stats_summary, "target_correlations": corr_matrix}
