# Lens v2.2 Benchmark Reproduction
FROM node:20-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node dependencies
RUN npm ci --only=production

# Install Python dependencies for evaluation
RUN pip3 install numpy scipy scikit-learn

# Copy application code
COPY . .

# Copy corpus and evaluation data
COPY corpus/ ./corpus/
COPY queries/ ./queries/
COPY embeddings.bin ./embeddings.bin
COPY golden_dataset.json ./golden_dataset.json

# Set environment variables
ENV NODE_ENV=reproduction
ENV FINGERPRINT=v22_1f3db391_1757345166574
ENV SLA_MS=150
ENV BOOTSTRAP_SAMPLES=2000

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run benchmark reproduction
CMD ["npm", "run", "benchmark:repro"]
