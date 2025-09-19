# Lens - Production Code Search Engine
# Complete build system with real implementations (no simulation)

.PHONY: help build test clean install dev lint format check bench docs docker

# Default target
help: ## Show this help message
	@echo "Lens - Production Code Search Engine"
	@echo "======================================"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Build targets
build: ## Build all packages in release mode
	@echo "🔨 Building Lens (production mode)..."
	cargo build --release --workspace

build-debug: ## Build all packages in debug mode
	@echo "🔨 Building Lens (debug mode)..."
	cargo build --workspace

# Individual package builds
build-search-engine: ## Build search engine package
	@echo "🔍 Building search engine..."
	cargo build --package lens-search-engine --release

build-lsp-server: ## Build LSP server package
	@echo "🔧 Building LSP server..."
	cargo build --package lens-lsp-server --release

build-lens: ## Build main Lens application
	@echo "🚀 Building Lens application..."
	cargo build --package lens --release

# Testing targets
test: ## Run all tests
	@echo "🧪 Running all tests..."
	cargo test --workspace

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	cargo test --workspace --lib

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	cargo test --workspace --test '*'

test-search-engine: ## Test search engine package
	@echo "🔍 Testing search engine..."
	cargo test --package lens-search-engine

test-lsp-server: ## Test LSP server package
	@echo "🔧 Testing LSP server..."
	cargo test --package lens-lsp-server

test-lens: ## Test main Lens application
	@echo "🚀 Testing Lens application..."
	cargo test --package lens

# Code quality targets
lint: ## Run clippy linter
	@echo "📝 Running clippy..."
	cargo clippy --workspace --all-targets --all-features -- -D warnings

format: ## Format code with rustfmt
	@echo "🎨 Formatting code..."
	cargo fmt --all

format-check: ## Check code formatting
	@echo "🎨 Checking code formatting..."
	cargo fmt --all -- --check

check: ## Check code without building
	@echo "✅ Checking code..."
	cargo check --workspace --all-targets --all-features

# Benchmarking
bench: ## Run benchmarks
	@echo "⚡ Running benchmarks..."
	cargo bench --workspace

bench-search: ## Benchmark search engine
	@echo "🔍 Benchmarking search engine..."
	cargo bench --package lens-search-engine

# Installation targets
install: build ## Install Lens binary
	@echo "📦 Installing Lens..."
	cargo install --path apps/lens-core --force

install-dev: build-debug ## Install Lens binary (debug)
	@echo "📦 Installing Lens (debug)..."
	cargo install --path apps/lens-core --debug --force

# Development targets
dev: ## Run Lens in development mode with auto-reload
	@echo "🔄 Starting development server..."
	cargo watch -x "run --package lens -- serve --cors"

dev-lsp: ## Run LSP server in development mode
	@echo "🔧 Starting LSP server..."
	cargo run --package lens -- lsp

dev-index: ## Index current directory for development
	@echo "📚 Indexing current directory..."
	cargo run --package lens -- index .

dev-search: ## Interactive search for development
	@echo "🔍 Starting interactive search..."
	@read -p "Enter search query: " query; \
	cargo run --package lens -- search "$$query" --limit 20

# Cleaning targets
clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	cargo clean

clean-index: ## Clean search index
	@echo "🧹 Cleaning search index..."
	rm -rf ./index
	rm -rf ./indexed-content

clean-all: clean clean-index ## Clean everything
	@echo "🧹 Cleaning everything..."
	rm -rf target/
	rm -rf */target/
	rm -rf ./*.log

# Documentation targets
docs: ## Generate documentation
	@echo "📖 Generating documentation..."
	cargo doc --workspace --no-deps --document-private-items

docs-open: docs ## Generate and open documentation
	@echo "📖 Opening documentation..."
	cargo doc --workspace --no-deps --document-private-items --open

# Docker targets
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t lens:latest .

docker-run: docker-build ## Build and run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p 3000:3000 -p 9999:9999 lens:latest

# Security and audit
audit: ## Run security audit
	@echo "🔒 Running security audit..."
	cargo audit

deny: ## Run cargo-deny checks
	@echo "🚫 Running cargo-deny..."
	cargo deny check

# Real-world workflow targets
setup: ## Set up development environment
	@echo "🚀 Setting up development environment..."
	@command -v rustc >/dev/null 2>&1 || { echo "❌ Rust not installed. Please install from https://rustup.rs/"; exit 1; }
	@command -v cargo >/dev/null 2>&1 || { echo "❌ Cargo not found"; exit 1; }
	rustup component add clippy rustfmt
	cargo install cargo-watch cargo-audit cargo-deny
	@echo "✅ Development environment ready!"

setup-index: install ## Set up and create initial index
	@echo "📚 Setting up search index..."
	mkdir -p ./indexed-content
	./target/release/lens index ./src --progress
	@echo "✅ Index created!"

demo: setup-index ## Run a complete demo
	@echo "🎬 Running Lens demo..."
	@echo "Searching for 'SearchEngine'..."
	./target/release/lens search "SearchEngine" --limit 5
	@echo ""
	@echo "Searching for function symbols..."
	./target/release/lens search "fn " --symbols --limit 5
	@echo ""
	@echo "Showing index stats..."
	./target/release/lens stats

# Production targets
release: clean test lint ## Full release build with checks
	@echo "🚀 Creating release build..."
	cargo build --release --workspace
	@echo "✅ Release build complete!"

release-package: release ## Create release package
	@echo "📦 Creating release package..."
	mkdir -p release/lens-$(shell cargo metadata --format-version 1 | jq -r '.workspace_members[0]' | cut -d' ' -f2)
	cp target/release/lens release/lens-$(shell cargo metadata --format-version 1 | jq -r '.workspace_members[0]' | cut -d' ' -f2)/
	cp README.md release/lens-$(shell cargo metadata --format-version 1 | jq -r '.workspace_members[0]' | cut -d' ' -f2)/
	cd release && tar -czf lens-$(shell cargo metadata --format-version 1 | jq -r '.workspace_members[0]' | cut -d' ' -f2).tar.gz lens-$(shell cargo metadata --format-version 1 | jq -r '.workspace_members[0]' | cut -d' ' -f2)
	@echo "✅ Release package: release/lens-*.tar.gz"

# Performance testing
perf-test: build ## Run performance tests
	@echo "⚡ Running performance tests..."
	./target/release/lens index ./tests/fixtures --progress
	time ./target/release/lens search "function" --limit 100
	./target/release/lens stats

# Integration testing
integration-test: build ## Run integration tests
	@echo "🔗 Running integration tests..."
	./scripts/integration-test.sh

# LSP testing
test-lsp: build ## Test LSP server functionality
	@echo "🔧 Testing LSP server..."
	./scripts/test-lsp.sh

# End-to-end testing
e2e-test: build ## Run end-to-end tests
	@echo "🎯 Running E2E tests..."
	./scripts/e2e-test.sh

# Monitoring and health checks
health-check: ## Run health checks on built binary
	@echo "💓 Running health checks..."
	./target/release/lens stats || echo "❌ Binary not working"
	@echo "✅ Health check complete"

# Database/Index management
backup-index: ## Backup search index
	@echo "💾 Backing up search index..."
	tar -czf index-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz ./index

restore-index: ## Restore search index from backup
	@echo "📥 Restoring search index..."
	@read -p "Enter backup file name: " backup; \
	tar -xzf "$$backup"

# CI/CD simulation
ci: clean test lint format-check ## Simulate CI pipeline
	@echo "🔄 Running CI pipeline..."
	@echo "✅ All checks passed!"

cd: release ## Simulate CD pipeline
	@echo "🚀 Running CD pipeline..."
	@echo "✅ Deployment ready!"

# Development utilities
watch-test: ## Watch files and run tests on changes
	@echo "👀 Watching for changes..."
	cargo watch -x "test --workspace"

watch-build: ## Watch files and build on changes
	@echo "👀 Watching for changes..."
	cargo watch -x "build --workspace"

# Language Server Protocol utilities
lsp-test-client: ## Run LSP test client
	@echo "🔧 Running LSP test client..."
	cd packages/lsp-server && cargo run --example test_client

# Search engine utilities
index-benchmark: ## Benchmark indexing performance
	@echo "📊 Benchmarking indexing..."
	time ./target/release/lens index ./src --force --progress

search-benchmark: ## Benchmark search performance
	@echo "📊 Benchmarking search..."
	./scripts/search-benchmark.sh

# Code coverage
coverage: ## Generate code coverage report
	@echo "📊 Generating coverage report..."
	cargo tarpaulin --all-features --workspace --ignore-tests --timeout 900 --out Html --output-dir coverage

# Dependency management
update-deps: ## Update dependencies
	@echo "📦 Updating dependencies..."
	cargo update

check-deps: ## Check dependency licenses and vulnerabilities
	@echo "🔍 Checking dependencies..."
	cargo audit
	cargo deny check licenses

# Documentation and examples
examples: ## Run all examples
	@echo "📚 Running examples..."
	cargo run --example basic_search
	cargo run --example lsp_integration

# Final verification
verify: clean ci ## Complete verification pipeline
	@echo "✅ Full verification complete!"

# Emergency procedures
emergency-rebuild: clean-all setup build test ## Emergency full rebuild
	@echo "🚨 Emergency rebuild complete!"

# Version information
version: ## Show version information
	@echo "Lens Version Information:"
	@echo "========================"
	@cargo metadata --format-version 1 | jq -r '.workspace_members[]' | while read member; do \
		name=$$(echo $$member | cut -d' ' -f1); \
		version=$$(echo $$member | cut -d' ' -f2); \
		echo "$$name: $$version"; \
	done