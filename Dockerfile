# Multi-stage Docker build for Lens
FROM node:20-alpine AS builder

# Install system dependencies for native modules
RUN apk add --no-cache \
    build-base \
    python3 \
    make \
    g++

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY src ./src

# Build TypeScript
RUN npm run build

# Production stage
FROM node:20-alpine AS runtime

# Install runtime dependencies
RUN apk add --no-cache \
    dumb-init \
    curl

# Create non-root user
RUN addgroup -g 1001 -S lens && \
    adduser -S lens -u 1001

WORKDIR /app

# Copy built application
COPY --from=builder --chown=lens:lens /app/dist ./dist
COPY --from=builder --chown=lens:lens /app/node_modules ./node_modules
COPY --chown=lens:lens package*.json ./

# Create data directories
RUN mkdir -p /app/data /app/segments /app/logs && \
    chown -R lens:lens /app

# Switch to non-root user
USER lens

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Expose ports
EXPOSE 3000 9464

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/server.js"]