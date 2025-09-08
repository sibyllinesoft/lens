# ast-grep - Tree-sitter structural search with JSON API wrapper  
FROM rust:1.75-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install ast-grep from GitHub
RUN cargo install ast-grep --locked

# Build the API wrapper
COPY Cargo.toml .
COPY server.rs src/main.rs

RUN cargo build --release

# Runtime image  
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ast-grep binary
COPY --from=builder /usr/local/cargo/bin/ast-grep /usr/local/bin/ast-grep

# Copy API server
COPY --from=builder /app/target/release/ast-grep-server /usr/local/bin/ast-grep-server

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8082/health || exit 1

ENV AST_GREP_SERVER_PORT=8082
ENV CORPUS_PATH=/datasets

EXPOSE 8082

CMD ["ast-grep-server"]