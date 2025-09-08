# livegrep - Codesearch-based real-time grep
FROM golang:1.21-alpine as builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git build-base

# Clone livegrep from GitHub
RUN git clone https://github.com/livegrep/livegrep.git .

# Build livegrep
RUN make GITHUB_CHECKOUT=1

# Runtime image
FROM alpine:3.18

RUN apk add --no-cache ca-certificates libc6-compat

WORKDIR /app

# Copy binaries
COPY --from=builder /app/bin/livegrep /usr/local/bin/livegrep
COPY --from=builder /app/bin/codesearch /usr/local/bin/codesearch

# Copy web assets
COPY --from=builder /app/web /app/web

# Create directories
RUN mkdir -p /index /datasets

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD wget -q --spider http://localhost:9898/ || exit 1

ENV LIVEGREP_LISTEN=:9898
ENV LIVEGREP_INDEX_DIR=/index

EXPOSE 9898

# Start livegrep server
CMD ["livegrep", "-addr", ":9898", "-docroot", "/app/web", "-index", "/index"]