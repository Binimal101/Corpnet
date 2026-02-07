.PHONY: install dev test lint format serve migrate seed clean docker-up docker-down

# Development setup
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Linting
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

# Server
serve:
	uvicorn src.api.server:app --reload --port 8000

# Database
migrate:
	python scripts/migrate_db.py

seed:
	python scripts/seed_data.py

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Peer nodes
peer:
	python -m src.api.cli peer --peer-id peer-001

super-peer:
	python -m src.api.cli peer --peer-id super-001 --super-peer

# Load testing
load-test:
	python scripts/load_test.py --requests 100 --concurrency 10

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
