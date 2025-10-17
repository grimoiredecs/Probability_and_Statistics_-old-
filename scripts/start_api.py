#!/usr/bin/env python3
"""
CLI Runner for starting the production FastAPI Inference & Ingestion Web Server.
"""

import uvicorn


def start():
    print("Launching Hardware Benchmarking FastAPI Server on port 8000...")
    uvicorn.run("src.hardware_benchmarking.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
