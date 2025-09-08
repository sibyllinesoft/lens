FROM node:20-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for data processing
RUN pip3 install pandas matplotlib seaborn numpy scipy

# Copy package files
COPY package*.json ./
RUN npm install --production

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8087/health || exit 1

EXPOSE 8087

CMD ["node", "publisher.js"]