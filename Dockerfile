FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANONYMIZED_TELEMETRY=False \
    GRADIO_ANALYTICS_ENABLED=False \
    FASTEMBED_CACHE_PATH=/app/models

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY chroma ./chroma
COPY cache ./cache

# Download the embedding model at build time so the container never has to at startup.
RUN python -c "from src.embeddings import get_embedder; get_embedder()"

# Hosts pass the port in $PORT.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
