# Milestone 4 — reproducible container for the generative-explanation stage.
#
# Build:  docker build -t alz-nlp .
# Run:    docker run --rm -v "$(pwd)/outputs:/app/outputs" alz-nlp
#
# The default CMD runs `python src/model_runner.py`, which generates the
# explanation samples into /app/outputs (mounted above so they appear on the
# host). No credentialed MIMIC data is required — the runner falls back to the
# bundled synthetic notes.

FROM python:3.11-slim

# Avoid interactive prompts and keep the image lean.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the project.
COPY . .

# Generate explanations by default. Override with e.g.
#   docker run --rm alz-nlp python -m src.run_pipeline --no-mount
CMD ["python", "src/model_runner.py"]
