FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by WeasyPrint runtime on Cloud Run.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    shared-mime-info \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY storage/ ./storage/
COPY market_data/ ./market_data/

# Run as non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Cloud Run sets PORT environment variable
ENV PORT=8080
EXPOSE 8080

# Use exec form to properly handle signals
CMD exec gunicorn src.api.server:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 3600 \
    --graceful-timeout 3600 \
    --keep-alive 75 \
    --workers 1 \
    --bind 0.0.0.0:$PORT
