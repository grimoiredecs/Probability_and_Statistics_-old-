import pandas as pd

from src.hardware_benchmarking.infrastructure.data.cleaning import DataCleaner


def test_cleaner_normalizes_specs_and_leaves_feature_imputation_for_pipeline():
    source = pd.DataFrame({
        "Processor_Base_Frequency": ["3,200 MHz", "N/A", "3.2 GHz", "3.2 GHz"],
        "Max_Turbo_Frequency": ["4.0 GHz", None, "4.0 GHz", "4.0 GHz"],
        "Vertical_Segment": [" Desktop ", "Desktop", "Desktop", "Desktop"],
        "nb_of_Cores": [4, 4, 4, 4],
        "TDP": ["65 W", "65 W", "-1 W", "65 W"],
    })

    cleaned = DataCleaner().clean_cpu_dataset(source)

    assert cleaned["Processor_Base_Frequency"].tolist() == [3.2, 3.2]
    assert cleaned["Vertical_Segment"].tolist() == ["Desktop", "Desktop"]
    assert cleaned["TDP"].isna().sum() == 1
