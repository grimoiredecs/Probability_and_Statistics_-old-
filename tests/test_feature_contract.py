import pandas as pd

from src.hardware_benchmarking.infrastructure.data.feature_store import FeatureStore


def test_gpu_features_never_include_target_derived_ratio(tmp_path):
    raw = pd.DataFrame({
        "Core_Speed": [1000.0], "Memory_Speed": [1500.0], "Max_Power": [100.0],
        "Memory_Bandwidth": [200.0], "ROPs": [20.0], "TMUs": [40.0],
    })
    features = FeatureStore(store_dir=str(tmp_path)).engineer_gpu_features(raw)

    assert "Core_to_Mem_Speed_Ratio" not in features
    assert "Core_to_Mem_Speed_Ratio" not in FeatureStore.GPU_FEATURE_COLS
