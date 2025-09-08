# Benchmark Coordinator - Protocol v2.0 Matrix Execution
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    netcat-traditional \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy coordinator code
COPY coordinator.py .
COPY scenario_matrix.py .
COPY statistical_analysis.py .
COPY adapters/ ./adapters/
COPY scripts/ ./scripts/

# Create directories
RUN mkdir -p /datasets /results /config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8085/health || exit 1

ENV PROTOCOL_VERSION=v2.0
ENV SLA_TIMEOUT_MS=150
ENV BENCHMARK_RUNS=1000
ENV STATISTICAL_CONFIDENCE=0.95
ENV BOOTSTRAP_SAMPLES=10000

EXPOSE 8085

CMD ["python", "coordinator.py"]