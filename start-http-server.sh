#!/bin/bash
# Start the Lens HTTP API server
# 
# This script starts the new Rust HTTP server that replaces the TypeScript Fastify server.
# The HTTP server provides full API compatibility while delivering better performance.

set -euo pipefail

# Configuration
DEFAULT_PORT=3000
DEFAULT_BIND="127.0.0.1"
DEFAULT_INDEX_PATH="./indexed-content"

# Parse command line arguments
PORT=${1:-$DEFAULT_PORT}
BIND_ADDR=${2:-$DEFAULT_BIND}
INDEX_PATH=${3:-$DEFAULT_INDEX_PATH}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Lens HTTP API Server${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""
echo -e "Server Address: ${GREEN}${BIND_ADDR}:${PORT}${NC}"
echo -e "Index Path: ${GREEN}${INDEX_PATH}${NC}"
echo -e "API Version: ${GREEN}v1${NC}"
echo ""

# Check if index exists
if [ ! -d "$INDEX_PATH" ]; then
    echo -e "${YELLOW}⚠️ Warning: Index directory '$INDEX_PATH' does not exist${NC}"
    echo -e "${YELLOW}   The server will start but searches may not work until an index is created.${NC}"
    echo ""
fi

# Check if Rust project is built
if [ ! -f "target/release/lens" ] && [ ! -f "target/debug/lens" ]; then
    echo -e "${YELLOW}📦 Building Rust project...${NC}"
    if cargo build --release; then
        echo -e "${GREEN}✅ Build successful${NC}"
    else
        echo -e "${RED}❌ Build failed${NC}"
        exit 1
    fi
    echo ""
fi

# Set environment variables for optimal performance
export RUST_LOG="${RUST_LOG:-lens_core=info,axum=info,tower=info}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

# Check if port is available
if command -v nc &> /dev/null; then
    if nc -z "$BIND_ADDR" "$PORT" 2>/dev/null; then
        echo -e "${RED}❌ Error: Port $PORT is already in use${NC}"
        echo -e "${YELLOW}   Try a different port or stop the existing service${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}🌟 Starting HTTP server...${NC}"
echo -e "${BLUE}   Endpoints available at: http://${BIND_ADDR}:${PORT}${NC}"
echo -e "${BLUE}   • GET  /health              - System health${NC}"
echo -e "${BLUE}   • GET  /manifest            - API information${NC}"
echo -e "${BLUE}   • POST /search              - Main search endpoint${NC}"
echo -e "${BLUE}   • POST /struct              - Structural search${NC}"
echo -e "${BLUE}   • POST /symbols/near        - Symbol proximity search${NC}"
echo -e "${BLUE}   • GET  /compat/check        - Version compatibility${NC}"
echo -e "${BLUE}   • Many more SPI and LSP endpoints...${NC}"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server
if [ -f "target/release/lens" ]; then
    exec ./target/release/lens serve \
        --bind "$BIND_ADDR" \
        --port "$PORT" \
        --index-path "$INDEX_PATH" \
        --enable-lsp true \
        --cache-ttl 24
else
    exec ./target/debug/lens serve \
        --bind "$BIND_ADDR" \
        --port "$PORT" \
        --index-path "$INDEX_PATH" \
        --enable-lsp true \
        --cache-ttl 24
fi