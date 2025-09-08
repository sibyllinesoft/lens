# Lens Core System - Benchmark Protocol v2.0 Compatible
FROM rust:1.75-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Build the application
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/lens /usr/local/bin/lens

# Copy configuration
COPY config/ /app/config/

# Create directories
RUN mkdir -p /app/index /results /datasets

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:50051/health || exit 1

# Set environment for benchmark mode
ENV RUST_LOG=info
ENV LENS_MODE=benchmark-v2
ENV BENCHMARK_PROTOCOL=v2.0

EXPOSE 50051

CMD ["lens", "--config", "/app/config/benchmark-v2.toml"]