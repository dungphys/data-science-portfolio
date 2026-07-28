FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffers stdout (cleaner container logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential: some scientific-python deps (e.g. scipy/neurokit2 transitive deps)
# fall back to source builds on certain platforms; curl: used by the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code
COPY app.py ecg_pipeline.py ./
COPY .streamlit ./.streamlit

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]