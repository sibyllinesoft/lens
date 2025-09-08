# Lens v2.2 External Reproduction Makefile

# Configuration
FINGERPRINT=v22_1f3db391_1757345166574
EXPECTED_TOLERANCE=0.001
DOCKER_COMPOSE=docker-compose -f docker/docker-compose.yml

.PHONY: repro setup build run validate clean help

# Main reproduction target
repro: setup build run validate
	@echo "🎉 Reproduction complete! Check results/hero_span_v22.csv"

# Setup corpus and dependencies
setup:
	@echo "📦 Setting up reproduction environment..."
	chmod +x scripts/download-corpus.sh
	./scripts/download-corpus.sh
	@echo "✅ Corpus download complete"

# Build Docker images
build:
	@echo "🔨 Building Docker images..."
	$(DOCKER_COMPOSE) build
	@echo "✅ Docker build complete"

# Run benchmark reproduction
run:
	@echo "🚀 Running benchmark reproduction..."
	@echo "⏱️  Expected duration: 45-60 minutes"
	@echo "🎯 Target: ±0.1 pp tolerance vs published results"
	mkdir -p results logs
	$(DOCKER_COMPOSE) up -d postgres redis
	@echo "⏳ Waiting for services to start..."
	sleep 30
	$(DOCKER_COMPOSE) run --rm lens-benchmark
	@echo "✅ Benchmark execution complete"

# Validate reproduction results
validate:
	@echo "🔍 Validating reproduction results..."
	node scripts/validate-reproduction.js
	@echo "📊 Validation complete - check validation report"

# Clean up environment
clean:
	@echo "🧹 Cleaning up..."
	$(DOCKER_COMPOSE) down -v
	docker system prune -f
	rm -rf results/*.json logs/*.log
	@echo "✅ Cleanup complete"

# Development targets
dev-setup:
	@echo "🛠️  Setting up development environment..."
	npm install
	pip3 install -r requirements.txt

logs:
	$(DOCKER_COMPOSE) logs -f lens-benchmark

shell:
	$(DOCKER_COMPOSE) exec lens-benchmark /bin/bash

health:
	@echo "🏥 Checking service health..."
	curl -f http://localhost:3000/health || echo "❌ Service not healthy"
	$(DOCKER_COMPOSE) ps

# Help
help:
	@echo "Lens v2.2 Reproduction Makefile"
	@echo ""
	@echo "Main targets:"
	@echo "  repro     - Complete reproduction (setup + build + run + validate)"
	@echo "  setup     - Download corpus and setup environment"  
	@echo "  build     - Build Docker images"
	@echo "  run       - Execute benchmark reproduction"
	@echo "  validate  - Validate results against expected values"
	@echo "  clean     - Clean up environment and temporary files"
	@echo ""
	@echo "Development targets:"
	@echo "  dev-setup - Setup development dependencies"
	@echo "  logs      - Follow application logs" 
	@echo "  shell     - Access container shell"
	@echo "  health    - Check service health"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Expected results file: results/hero_span_v22.csv"
	@echo "Tolerance: ±0.1 pp nDCG@10"
	@echo "Duration: ~60 minutes end-to-end"
