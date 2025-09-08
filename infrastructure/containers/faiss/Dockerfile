# FAISS Server - Pure ANN library wrapper for benchmarking
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libopenblas-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install FAISS from source (latest version)
RUN git clone https://github.com/facebookresearch/faiss.git /tmp/faiss && \
    cd /tmp/faiss && \
    cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON && \
    make -C build -j$(nproc) && \
    make -C build install && \
    cd build/faiss/python && \
    pip install .

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime image
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy FAISS installation
COPY --from=builder /usr/local/lib/python3.11/site-packages/faiss* /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/lib/libfaiss* /usr/local/lib/
COPY --from=builder /app /app

# Copy server code
COPY server.py .
COPY vector_utils.py .

# Create directories
RUN mkdir -p /indices /datasets /results

# Health check
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8084/health || exit 1

ENV FAISS_SERVER_PORT=8084
ENV INDEX_TYPE=IVF_HNSW
ENV VECTOR_DIM=512
ENV PYTHONPATH=/app

EXPOSE 8084

CMD ["python", "server.py"]