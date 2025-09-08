# comby - Structural search and rewrite with JSON API wrapper
FROM ocaml/opam:ubuntu-22.04 as builder

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libpcre3-dev \
    libev-dev \
    m4 \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER opam

# Install comby via opam
RUN opam update && opam install -y comby

# Install Node.js for API wrapper
USER root
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Create API wrapper
WORKDIR /app
COPY package.json .
COPY server.js .

RUN npm install

# Runtime image
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libpcre3 \
    libev4 \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy comby binary
COPY --from=builder /home/opam/.opam/default/bin/comby /usr/local/bin/comby

# Copy API server
COPY --from=builder /app /app

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8081/health || exit 1

ENV COMBY_SERVER_PORT=8081
ENV CORPUS_PATH=/datasets

EXPOSE 8081

CMD ["node", "server.js"]