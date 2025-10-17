# Senior Developer Production Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv

# Copy dependency definition
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN /uv/bin/uv sync --frozen --no-install-project

# Copy project source code
COPY . .

# Environment path
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 5000

# Default command: launch production FastAPI inference & ingestion server
CMD ["uvicorn", "src.hardware_benchmarking.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
