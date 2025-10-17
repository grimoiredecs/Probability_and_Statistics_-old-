import pytest
from fastapi.testclient import TestClient
from src.hardware_benchmarking.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "HEALTHY"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200


def test_portfolio_page():
    response = client.get("/portfolio")
    assert response.status_code == 200
    assert "Hardware benchmarking" in response.text
    assert "CPU CHAMPION" in response.text


def test_cpu_prediction_endpoint():
    payload = {
        "nb_of_Cores": 8.0,
        "nb_of_Threads": 16.0,
        "TDP": 65.0,
        "Cache": 16.0,
        "Lithography": 14.0,
        "Vertical_Segment": "Desktop",
    }
    response = client.post("/predict/cpu", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["domain"] == "CPU"
    assert "predicted_base_frequency_ghz" in json_data
    assert json_data["predicted_base_frequency_ghz"] > 0.0
