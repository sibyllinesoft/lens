# Lens v2.2 Reproduction Makefile

repro: build run validate

build:
	@echo "🔨 Building reproduction environment..."
	docker compose build

run:
	@echo "🚀 Running benchmark reproduction..."
	docker compose up -d
	@echo "⏳ Waiting for services to be ready..."
	sleep 30
	docker compose exec lens npm run benchmark:repro

validate:
	@echo "✅ Validating reproduction results..."
	docker compose exec lens npm run validate:repro

clean:
	docker compose down -v

.PHONY: repro build run validate clean
