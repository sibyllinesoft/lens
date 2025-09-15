# Lens Production Docker Image
FROM node:20-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy essential files for the lens application
COPY package.json ./
COPY src ./src/
COPY scripts ./scripts/
COPY target ./target/
COPY manifests ./manifests/
COPY configs ./configs/

# Set environment variables
ENV NODE_ENV=production
ENV RUST_LOG=info

# Expose port
EXPOSE 3000 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/health || curl -f http://localhost:8000/health || exit 1

# Default command - run the existing lens binary if available, otherwise fallback
CMD ["sh", "-c", "if [ -f ./target/release/lens ]; then ./target/release/lens; else node src/server.js; fi"]