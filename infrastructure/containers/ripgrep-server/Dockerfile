# Ripgrep Server - JSON API wrapper for benchmarking
FROM rust:1.75-slim as builder

WORKDIR /app

# Install ripgrep and build dependencies  
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install ripgrep binary
RUN cargo install ripgrep

# Create the server source
COPY server.rs src/main.rs
COPY Cargo.toml .

# Build the API server
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ripgrep binary
COPY --from=builder /usr/local/cargo/bin/rg /usr/local/bin/rg

# Copy API server
COPY --from=builder /app/target/release/ripgrep-server /usr/local/bin/ripgrep-server

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

ENV RIPGREP_SERVER_PORT=8080
ENV CORPUS_PATH=/datasets

EXPOSE 8080

CMD ["ripgrep-server"]