# System Monitor - Hardware attestation and SLA measurement
FROM golang:1.21-alpine as builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git build-base

# Copy source code
COPY go.mod go.sum ./
COPY *.go ./

# Build the monitor
RUN go mod download
RUN go build -o monitor .

# Runtime image with monitoring tools
FROM alpine:3.18

# Install monitoring and attestation tools
RUN apk add --no-cache \
    ca-certificates \
    curl \
    procps \
    util-linux \
    dmidecode \
    lscpu \
    lshw \
    pciutils \
    usbutils \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy monitor binary
COPY --from=builder /app/monitor /usr/local/bin/monitor

# Copy configuration and scripts
COPY config/ /app/config/
COPY scripts/ /app/scripts/

# Create directories
RUN mkdir -p /results /attestation

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:9090/health || exit 1

ENV MONITOR_INTERVAL=1s
ENV ATTESTATION_MODE=enabled
ENV RESULTS_DIR=/results

EXPOSE 9090

CMD ["monitor", "--config", "/app/config/monitor.yaml"]