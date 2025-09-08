# Multi-stage Dockerfile for fraud-resistant builds
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY proto/ ./proto/

# Build dependencies (this is the caching layer)
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -f target/release/deps/lens_core*

# Copy source and build
COPY src/ ./src/
COPY benches/ ./benches/

# Build with attestation info
ARG GIT_SHA=unknown
ARG BUILD_TIMESTAMP
ENV GIT_SHA=${GIT_SHA}
ENV BUILD_TIMESTAMP=${BUILD_TIMESTAMP}
ENV LENS_MODE=real

RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /app/target/release/lens-core /usr/local/bin/lens-core

# Health check with mode verification
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:50051/health || exit 1

EXPOSE 50051

# Ensure real mode
ENV LENS_MODE=real

CMD ["lens-core"]