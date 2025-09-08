#!/bin/bash
# Deploy Benchmark Protocol v2.0 Infrastructure
# Comprehensive deployment script for authentic competitor benchmarking

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose-v2.yml"
LOG_FILE="/tmp/benchmark-v2-deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for Protocol v2.0 deployment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available disk space (need at least 50GB)
    AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 50 ]; then
        warning "Available disk space is less than 50GB. Benchmarking may fail due to insufficient storage."
    fi
    
    # Check available memory (need at least 16GB)
    AVAILABLE_MEMORY=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$AVAILABLE_MEMORY" -lt 16 ]; then
        warning "Available memory is less than 16GB. Some systems may not start properly."
    fi
    
    success "Prerequisites check completed"
}

# Generate environment configuration
generate_environment() {
    log "Generating environment configuration..."
    
    cat > "$PROJECT_ROOT/.env" << EOF
# Benchmark Protocol v2.0 Environment Configuration
# Generated on: $(date -Iseconds)

# Git information
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
BUILD_TIMESTAMP=$(date -Iseconds)

# System URLs (Docker Compose internal networking)
LENS_URL=http://lens-core:50051
ZOEKT_URL=http://zoekt-webserver:6070
LIVEGREP_URL=http://livegrep:9898
RIPGREP_URL=http://ripgrep-server:8080
COMBY_URL=http://comby-server:8081
AST_GREP_URL=http://ast-grep-server:8082
OPENSEARCH_URL=http://opensearch:9200
QDRANT_URL=http://qdrant:6333
VESPA_URL=http://vespa:8080
FAISS_URL=http://faiss-server:8084

# Benchmark configuration
PROTOCOL_VERSION=v2.0
SLA_TIMEOUT_MS=150
BENCHMARK_RUNS=1000
STATISTICAL_CONFIDENCE=0.95
BOOTSTRAP_SAMPLES=10000
MAX_CONCURRENT=10
WARMUP_QUERIES=5

# Hardware configuration
CPU_GOVERNOR=performance
NUMA_POLICY=local

# Monitoring configuration  
MONITOR_INTERVAL=1s
ATTESTATION_MODE=enabled
RESULTS_DIR=/results

# Auto-start configuration
AUTO_START_BENCHMARK=false
AUTO_PUBLISH_RESULTS=false
EOF

    success "Environment configuration generated"
}

# Set system performance settings
configure_system_performance() {
    log "Configuring system performance settings..."
    
    # Set CPU governor to performance (if available)
    if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
        if [ -w "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" ]; then
            echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
            success "Set CPU governor to performance mode"
        else
            warning "Cannot set CPU governor - insufficient permissions"
        fi
    else
        warning "CPU frequency scaling not available"
    fi
    
    # Increase file descriptor limits
    if command -v ulimit &> /dev/null; then
        ulimit -n 65536
        success "Increased file descriptor limits"
    fi
    
    # Configure swap
    if [ -f "/proc/sys/vm/swappiness" ] && [ -w "/proc/sys/vm/swappiness" ]; then
        echo 1 | sudo tee /proc/sys/vm/swappiness > /dev/null
        success "Reduced swappiness for better performance"
    fi
    
    success "System performance configuration completed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p "$PROJECT_ROOT/benchmark-results"
    mkdir -p "$PROJECT_ROOT/benchmark-datasets"
    mkdir -p "$PROJECT_ROOT/benchmark-attestation"
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Set proper permissions
    chmod 755 "$PROJECT_ROOT/benchmark-results"
    chmod 755 "$PROJECT_ROOT/benchmark-datasets" 
    chmod 755 "$PROJECT_ROOT/benchmark-attestation"
    
    success "Directories created successfully"
}

# Pull required images
pull_images() {
    log "Pulling required Docker images..."
    
    # Use docker-compose to pull images
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" build --parallel
    else
        docker compose -f "$COMPOSE_FILE" build --parallel
    fi
    
    success "Docker images pulled successfully"
}

# Build custom images
build_images() {
    log "Building custom Docker images..."
    
    # Build all services
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" build --parallel
    else
        docker compose -f "$COMPOSE_FILE" build --parallel
    fi
    
    success "Docker images built successfully"
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying Protocol v2.0 infrastructure..."
    
    # Start infrastructure services first
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" up -d \
            dataset-loader \
            system-monitor
    else
        docker compose -f "$COMPOSE_FILE" up -d \
            dataset-loader \
            system-monitor
    fi
    
    log "Waiting for infrastructure services to initialize..."
    sleep 30
    
    success "Infrastructure services deployed"
}

# Deploy competitor systems
deploy_competitor_systems() {
    log "Deploying competitor systems..."
    
    # Start all competitor systems
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" up -d \
            zoekt-webserver \
            zoekt-indexserver \
            livegrep \
            ripgrep-server \
            comby-server \
            ast-grep-server \
            opensearch \
            qdrant \
            vespa \
            faiss-server
    else
        docker compose -f "$COMPOSE_FILE" up -d \
            zoekt-webserver \
            zoekt-indexserver \
            livegrep \
            ripgrep-server \
            comby-server \
            ast-grep-server \
            opensearch \
            qdrant \
            vespa \
            faiss-server
    fi
    
    log "Waiting for competitor systems to start..."
    sleep 60
    
    success "Competitor systems deployed"
}

# Deploy Lens system
deploy_lens_system() {
    log "Deploying Lens system..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" up -d lens-core
    else
        docker compose -f "$COMPOSE_FILE" up -d lens-core
    fi
    
    log "Waiting for Lens system to start..."
    sleep 30
    
    success "Lens system deployed"
}

# Deploy benchmark coordinator
deploy_benchmark_coordinator() {
    log "Deploying benchmark coordinator..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" up -d benchmark-coordinator
    else
        docker compose -f "$COMPOSE_FILE" up -d benchmark-coordinator
    fi
    
    log "Waiting for benchmark coordinator to start..."
    sleep 20
    
    success "Benchmark coordinator deployed"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment health..."
    
    local failed_services=0
    local total_services=0
    
    # List of services to check
    services=(
        "lens-core:50051/health"
        "zoekt-webserver:6070/"
        "ripgrep-server:8080/health"
        "comby-server:8081/health"
        "ast-grep-server:8082/health"
        "opensearch:9200/_cluster/health"
        "qdrant:6333/health"
        "faiss-server:8084/health"
        "benchmark-coordinator:8085/health"
        "system-monitor:9090/health"
    )
    
    for service_url in "${services[@]}"; do
        total_services=$((total_services + 1))
        
        service_name=$(echo "$service_url" | cut -d':' -f1)
        
        log "Checking health of $service_name..."
        
        if curl -f -m 10 "http://$service_url" > /dev/null 2>&1; then
            success "✅ $service_name is healthy"
        else
            error "❌ $service_name is not responding"
            failed_services=$((failed_services + 1))
        fi
    done
    
    log "Health check summary: $((total_services - failed_services))/$total_services services healthy"
    
    if [ $failed_services -eq 0 ]; then
        success "🎉 All services are healthy and ready for benchmarking!"
        return 0
    else
        error "⚠️  $failed_services services failed health checks"
        return 1
    fi
}

# Generate hardware attestation
generate_attestation() {
    log "Generating hardware attestation..."
    
    # Trigger attestation generation
    if curl -f -m 30 "http://localhost:9090/attestation" > "$PROJECT_ROOT/benchmark-attestation/hardware-attestation.json"; then
        success "Hardware attestation generated"
    else
        warning "Failed to generate hardware attestation - continuing anyway"
    fi
}

# Show deployment status
show_deployment_status() {
    log "Protocol v2.0 Deployment Status:"
    
    echo ""
    echo "🚀 Benchmark Protocol v2.0 Infrastructure Deployed Successfully!"
    echo ""
    echo "📊 Access Points:"
    echo "  • Benchmark Coordinator:  http://localhost:8085"
    echo "  • System Monitor:         http://localhost:9090"
    echo "  • Lens Core:              http://localhost:50051"
    echo "  • Zoekt Search:           http://localhost:6070"
    echo "  • OpenSearch:             http://localhost:9200"
    echo "  • Qdrant:                 http://localhost:6333"
    echo ""
    echo "📁 Data Directories:"
    echo "  • Results:                $PROJECT_ROOT/benchmark-results/"
    echo "  • Datasets:               $PROJECT_ROOT/benchmark-datasets/"
    echo "  • Attestation:            $PROJECT_ROOT/benchmark-attestation/"
    echo "  • Logs:                   $PROJECT_ROOT/logs/"
    echo ""
    echo "🔬 Start Benchmarking:"
    echo "  • Manual:                 curl -X POST http://localhost:8085/start-benchmark"
    echo "  • Status:                 curl http://localhost:8085/status"
    echo ""
    echo "📋 Logs:"
    echo "  • Deployment:             $LOG_FILE"
    echo "  • Services:               docker-compose -f $COMPOSE_FILE logs -f"
    echo ""
}

# Cleanup function
cleanup_on_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error "Deployment failed with exit code $exit_code"
        log "Cleaning up partially deployed services..."
        
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" down || true
        else
            docker compose -f "$COMPOSE_FILE" down || true
        fi
    fi
}

# Main deployment function
main() {
    log "Starting Protocol v2.0 deployment..."
    
    # Set trap for cleanup on error
    trap cleanup_on_error EXIT
    
    # Deployment steps
    check_prerequisites
    generate_environment
    configure_system_performance
    create_directories
    pull_images
    build_images
    deploy_infrastructure
    deploy_competitor_systems
    deploy_lens_system
    deploy_benchmark_coordinator
    
    # Wait for all services to fully start
    log "Waiting for all services to fully initialize..."
    sleep 60
    
    # Verification and attestation
    if verify_deployment; then
        generate_attestation
        show_deployment_status
        success "🎉 Protocol v2.0 deployment completed successfully!"
    else
        error "❌ Deployment completed with errors - some services may not be functional"
        exit 1
    fi
    
    # Disable cleanup trap on success
    trap - EXIT
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi