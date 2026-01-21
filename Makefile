# Dirghayu Makefile
# Quick commands for common development tasks

.PHONY: help install install-uv install-dev data demo api test lint format clean

help:  ## Show this help message
	@echo "Dirghayu Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with uv (fast!)
	uv pip install -r requirements.txt

install-pip:  ## Install dependencies with pip (slower)
	pip install -r requirements.txt

install-dev:  ## Install with dev dependencies
	uv pip install -e ".[dev]"

data:  ## Download/create sample data
	python scripts/download_data.py

demo:  ## Run end-to-end demo
	python demo.py data/sample.vcf

api:  ## Start API server
	python src/api/server.py

test:  ## Run tests
	pytest tests/ -v

lint:  ## Check code quality
	ruff check src/ scripts/ demo.py

format:  ## Format code
	ruff format src/ scripts/ demo.py

clean:  ## Clean temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache

# Quick workflow commands
quick-start: install data demo  ## Install, download data, and run demo

dev-setup: install-dev data  ## Setup development environment
