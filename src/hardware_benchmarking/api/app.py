import os
import json
import uuid
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from .schemas import CPUPredictionRequest, GPUPredictionRequest
from .portfolio import portfolio_page
from ..config import load_settings
from ..application.pipeline_factory import build_pipeline
from ..infrastructure.data.validation import DataValidator

app = FastAPI(
    title="Hardware Benchmarking Production API",
    description="Mark Richards' Pipes & Filters Architecture Production REST API for real-time inference and ingestion triggers.",
    version="2.0.0",
)


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _write_run_record(run_id: str, payload: dict) -> None:
    directory = Path("outputs") / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "run.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(directory / "run.json")


def execute_background_pipeline(domain_key: str, run_id: str):
    settings = load_settings()
    domain = settings.domains[domain_key]
    try:
        _write_run_record(run_id, {"run_id": run_id, "domain": domain_key, "status": "RUNNING"})
        pipeline, context = build_pipeline(settings, domain, run_id=run_id)
        result = pipeline.execute(context)
        _write_run_record(run_id, {"run_id": run_id, "domain": domain_key, "status": "COMPLETED", "metrics": result.champion_metrics, "gate_passed": result.gate_passed})
    except Exception as exc:
        _write_run_record(run_id, {"run_id": run_id, "domain": domain_key, "status": "FAILED", "error": str(exc)})


@app.get("/health")
def health():
    return {"status": "HEALTHY", "architecture": "Mark Richards Pipes & Filters", "version": "2.0.0"}


@app.get("/portfolio", include_in_schema=False)
def portfolio():
    """Human-friendly, read-only summary for demos and portfolio review."""
    return portfolio_page()


@app.get("/metrics")
def metrics():
    registry_path = os.path.join("models", "registry.json")
    if not os.path.exists(registry_path):
        return {"status": "NO_MODELS_REGISTERED"}
    with open(registry_path, "r") as f:
        return json.load(f)


@app.post("/predict/cpu")
def predict_cpu(req: CPUPredictionRequest):
    model_path = os.path.join("models", "cpu_champion.joblib")
    prep_path = os.path.join("models", "cpu_preprocessor.joblib")

    if not (os.path.exists(model_path) and os.path.exists(prep_path)):
        raise HTTPException(status_code=404, detail="CPU Champion Model artifact not found.")

    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)

        raw_dict = req.model_dump()
        cores = raw_dict["nb_of_Cores"] if raw_dict["nb_of_Cores"] > 0 else 1.0
        threads = raw_dict["nb_of_Threads"] if raw_dict["nb_of_Threads"] > 0 else cores

        raw_dict["Max_nb_of_Memory_Channels"] = raw_dict.get("Max_nb_of_Memory_Channels", 2.0)
        raw_dict["Recommended_Customer_Price"] = raw_dict.get("Recommended_Customer_Price", 300.0)
        raw_dict["Max_Memory_Size"] = raw_dict.get("Max_Memory_Size", 64.0)
        raw_dict["Max_Memory_Bandwidth"] = raw_dict.get("Max_Memory_Bandwidth", 41.6)

        raw_dict["Cache_per_Core"] = raw_dict["Cache"] / cores
        raw_dict["TDP_per_Core"] = raw_dict["TDP"] / cores
        raw_dict["TDP_per_Thread"] = raw_dict["TDP"] / threads
        raw_dict["Price_per_Core"] = raw_dict["Recommended_Customer_Price"] / cores
        raw_dict["Bandwidth_per_Core"] = raw_dict["Max_Memory_Bandwidth"] / cores
        raw_dict["Cores_per_Thread_Ratio"] = cores / threads

        input_df = pd.DataFrame([raw_dict])
        proc_X = preprocessor.transform(input_df)
        predicted_freq = float(model.predict(proc_X)[0])

        return {
            "domain": "CPU",
            "predicted_base_frequency_ghz": round(predicted_freq, 3),
            "input_specs": raw_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/gpu")
def predict_gpu(req: GPUPredictionRequest):
    model_path = os.path.join("models", "gpu_champion.joblib")
    prep_path = os.path.join("models", "gpu_preprocessor.joblib")

    if not (os.path.exists(model_path) and os.path.exists(prep_path)):
        raise HTTPException(status_code=404, detail="GPU Champion Model artifact not found.")

    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)

        raw_dict = req.model_dump()
        power = raw_dict["Max_Power"] if raw_dict["Max_Power"] > 0 else 1.0
        tmus = raw_dict["TMUs"] if raw_dict["TMUs"] > 0 else 1.0

        raw_dict["Bandwidth_per_Watt"] = raw_dict["Memory_Bandwidth"] / power
        raw_dict["ROPs_to_TMUs_Ratio"] = raw_dict["ROPs"] / tmus
        input_df = pd.DataFrame([raw_dict])
        proc_X = preprocessor.transform(input_df)
        predicted_speed = float(model.predict(proc_X)[0])

        return {
            "domain": "GPU",
            "predicted_core_speed_mhz": round(predicted_speed, 2),
            "input_specs": raw_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/ingest")
async def ingest(
    background_tasks: BackgroundTasks,
    domain: str = "cpu",
    file: UploadFile = File(...)
):
    domain_key = domain.lower()
    if domain_key not in ["cpu", "gpu"]:
        raise HTTPException(status_code=400, detail="Domain must be 'cpu' or 'gpu'")

    contents = await file.read()
    if not contents or len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV upload must be between 1 byte and 50 MiB.")
    try:
        new_df = pd.read_csv(__import__("io").BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}")

    settings = load_settings()
    domain_settings = settings.domains[domain_key]
    target_filepath = domain_settings.raw_filepath
    existing_df = pd.read_csv(target_filepath)
    if set(new_df.columns) != set(existing_df.columns):
        raise HTTPException(status_code=422, detail="Uploaded CSV schema must exactly match the existing raw dataset.")
    validator = DataValidator()
    validation = validator.validate_cpu_data(new_df) if domain_key == "cpu" else validator.validate_gpu_data(new_df)
    if validation["status"] != "PASSED":
        raise HTTPException(status_code=422, detail=validation.get("error", "Data validation failed."))
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    # Replace atomically only after parsing and validation complete.
    target = Path(target_filepath)
    temporary = target.with_suffix(target.suffix + ".tmp")
    updated_df.to_csv(temporary, index=False)
    temporary.replace(target)

    run_id = uuid.uuid4().hex
    _write_run_record(run_id, {"run_id": run_id, "domain": domain_key, "status": "QUEUED", "records_ingested": len(new_df)})
    background_tasks.add_task(execute_background_pipeline, domain_key, run_id)

    return {
        "message": f"Successfully ingested {len(new_df)} new records for domain '{domain_key}'. Triggered automated Pipes & Filters retraining loop.",
        "records_ingested": len(new_df),
        "total_records_now": len(updated_df),
        "run_id": run_id,
    }
