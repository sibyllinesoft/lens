# Dataset Loader - Authentic dataset acquisition and preprocessing
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    unzip \
    tar \
    gzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset processing scripts
COPY dataset_processor.py .
COPY verify_datasets.py .

# Create directories
RUN mkdir -p /datasets /scripts

# Set environment variables
ENV DATASETS_DIR=/datasets
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import os; exit(0 if os.path.exists('/datasets/.ready') else 1)"

# Default command
CMD ["python", "dataset_processor.py"]