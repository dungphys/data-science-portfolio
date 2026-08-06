FROM python:3.14-slim

# System deps needed by matplotlib/scikit-learn wheels stay minimal since
# we use manylinux wheels; keep only what's required for a clean build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure output directories exist even if not mounted as volumes
RUN mkdir -p data models results/figures

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Default action: run the full pipeline (EDA -> train -> business report).
# Override with `docker run <image> pytest` etc. as needed.
CMD ["python", "scripts/run_eda.py"]