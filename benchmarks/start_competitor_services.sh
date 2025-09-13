#!/bin/bash

# Start Competitor Services for Lens Benchmarking
# Starts all Docker services and waits for them to be ready

set -e

echo "🚀 Starting Lens Competitor Services"
echo "===================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose >/dev/null 2>&1; then
    echo "❌ docker-compose is not installed or not in PATH."
    exit 1
fi

# Navigate to the benchmarks directory
cd "$(dirname "$0")"

echo "📁 Working directory: $(pwd)"

# Create corpus directory if it doesn't exist
if [ ! -d "./corpus" ]; then
    echo "📚 Creating corpus directory..."
    mkdir -p ./corpus
    echo "⚠️  Corpus directory is empty. Please add source code files to ./corpus/"
fi

# Start services in detached mode
echo "🐳 Starting Docker services..."
docker-compose up -d

echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Function to check service health
check_service() {
    local name=$1
    local url=$2
    local endpoint=$3
    
    echo -n "   Checking $name... "
    if curl -sf "$url$endpoint" >/dev/null 2>&1; then
        echo "✅"
        return 0
    else
        echo "❌"
        return 1
    fi
}

# Check each service
failed_services=0

check_service "Zoekt" "http://localhost:6070" "/" || ((failed_services++))
check_service "Livegrep" "http://localhost:9898" "/" || ((failed_services++))
check_service "Ripgrep" "http://localhost:8080" "/health" || ((failed_services++))
check_service "Comby" "http://localhost:8081" "/health" || ((failed_services++))
check_service "AST-grep" "http://localhost:8082" "/health" || ((failed_services++))
check_service "OpenSearch" "http://localhost:9200" "/_cluster/health" || ((failed_services++))
check_service "Qdrant" "http://localhost:6333" "/health" || ((failed_services++))
check_service "Vespa" "http://localhost:8080" "/ApplicationStatus" || ((failed_services++))
check_service "FAISS" "http://localhost:8084" "/health" || ((failed_services++))
check_service "Milvus" "http://localhost:9091" "/healthz" || ((failed_services++))
check_service "ctags" "http://localhost:8083" "/health" || ((failed_services++))

echo ""
echo "📊 Service Status Summary:"
echo "   Total services: 11"
echo "   Working services: $((11 - failed_services))"
echo "   Failed services: $failed_services"

if [ $failed_services -eq 0 ]; then
    echo "🎉 All services are ready!"
    echo ""
    echo "🔬 Run tests with: python3 test_docker_infrastructure.py"
    echo "🏃 Run benchmarks with: python3 code_search_rag_benchmark.py --config real_systems_config.yaml"
elif [ $failed_services -le 3 ]; then
    echo "⚠️  Most services are ready. Some services may need more time to initialize."
    echo ""
    echo "💡 Check service logs with: docker-compose logs [service_name]"
    echo "💡 Retry in a few minutes or restart failed services"
else
    echo "❌ Many services failed to start. Check Docker logs:"
    echo "   docker-compose logs"
    echo ""
    echo "🔧 To restart services: docker-compose down && docker-compose up -d"
    exit 1
fi

echo ""
echo "🛠️  Management Commands:"
echo "   View logs: docker-compose logs -f [service_name]"
echo "   Stop services: docker-compose down"
echo "   Restart: docker-compose restart [service_name]"
echo "   Check status: docker-compose ps"