.PHONY: help run api test docker-build docker-up clean

help:
	@echo "Available developer commands:"
	@echo "  make run          Execute Pipes & Filters pipeline"
	@echo "  make api          Launch FastAPI REST inference server"
	@echo "  make test         Execute pytest suite"
	@echo "  make docker-build Build Docker production image"
	@echo "  make docker-up    Run Docker Compose stack"

run:
	uv run python scripts/run_pipeline.py

api:
	uv run python scripts/start_api.py

test:
	uv run pytest tests/ -v

docker-build:
	docker build -t hardware-mlops:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf __pycache__ .pytest_cache outputs/temp_ingest_*.csv
