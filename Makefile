.PHONY: help install install-all install-group docker-build docker-up docker-down jupyter clean format lint test

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)LLM Finetuning Cookbook - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Poetry Installation Commands
install: ## Install core dependencies only
	@echo "$(BLUE)Installing core dependencies...$(NC)"
	poetry install

install-all: ## Install all dependencies (all optional groups)
	@echo "$(BLUE)Installing all dependencies...$(NC)"
	poetry install --with full-finetuning,peft,instruction-tuning,reasoning,dev

install-full-finetuning: ## Install dependencies for Full Fine-Tuning notebooks
	@echo "$(BLUE)Installing full-finetuning dependencies...$(NC)"
	poetry install --with full-finetuning

install-peft: ## Install dependencies for PEFT notebooks
	@echo "$(BLUE)Installing PEFT dependencies...$(NC)"
	poetry install --with peft

install-instruction-tuning: ## Install dependencies for Instruction Tuning notebooks
	@echo "$(BLUE)Installing instruction-tuning dependencies...$(NC)"
	poetry install --with instruction-tuning

install-reasoning: ## Install dependencies for Reasoning notebooks
	@echo "$(BLUE)Installing reasoning dependencies...$(NC)"
	poetry install --with reasoning

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	poetry install --with dev

# Docker Commands
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker-compose build

docker-up: ## Start Jupyter in Docker (all dependencies)
	@echo "$(BLUE)Starting Jupyter Lab in Docker...$(NC)"
	@echo "$(YELLOW)Access at: http://localhost:8888$(NC)"
	docker-compose up -d jupyter-all
	@echo "$(GREEN)Jupyter Lab is running!$(NC)"

docker-down: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f jupyter-all

docker-shell: ## Open shell in Docker container
	docker-compose exec jupyter-all /bin/bash

# Development Commands
jupyter: ## Start Jupyter Lab locally
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	@echo "$(YELLOW)Access at: http://localhost:8888$(NC)"
	poetry run jupyter lab --no-browser

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	poetry run black .
	poetry run isort .
	@echo "$(GREEN)Code formatted!$(NC)"

lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	poetry run flake8 .
	poetry run mypy .

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	poetry run pytest

clean: ## Clean up generated files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Cleanup complete!$(NC)"

# Environment Commands
lock: ## Update poetry.lock file
	@echo "$(BLUE)Updating lock file...$(NC)"
	poetry lock

update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	poetry update

shell: ## Open poetry shell
	poetry shell

# Setup Commands
setup-dev: install-all ## Complete development setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	poetry run pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

setup-minimal: install ## Minimal setup (core dependencies only)
	@echo "$(GREEN)Minimal setup complete!$(NC)"

# Notebook Commands
notebook-clean: ## Remove output from all notebooks
	@echo "$(BLUE)Cleaning notebook outputs...$(NC)"
	poetry run nbstripout **/*.ipynb
	@echo "$(GREEN)Notebooks cleaned!$(NC)"

