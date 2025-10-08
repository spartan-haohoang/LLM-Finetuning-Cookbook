#!/bin/bash

# Development Environment Setup Script
# This script sets up everything needed for development

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LLM Finetuning Cookbook - Dev Setup${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if running in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found!${NC}"
    echo -e "Please run this script from the repository root."
    exit 1
fi

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}\n"

# Check if Poetry is installed
echo -e "${BLUE}Checking Poetry installation...${NC}"
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}Poetry not found. Installing...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH
    export PATH="$HOME/.local/bin:$PATH"
    
    # Add to shell config
    if [ -f "$HOME/.bashrc" ]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    if [ -f "$HOME/.zshrc" ]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
    fi
    
    echo -e "${GREEN}✓ Poetry installed${NC}\n"
else
    echo -e "${GREEN}✓ Poetry found${NC}\n"
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
poetry install --with full-finetuning,peft,instruction-tuning,reasoning,dev

echo -e "${GREEN}✓ Dependencies installed${NC}\n"

# Set up pre-commit hooks
echo -e "${BLUE}Setting up pre-commit hooks...${NC}"
poetry run pre-commit install

echo -e "${GREEN}✓ Pre-commit hooks installed${NC}\n"

# Set up Jupyter kernel
echo -e "${BLUE}Setting up Jupyter kernel...${NC}"
poetry run python -m ipykernel install --user --name llm-finetuning --display-name "LLM Finetuning"

echo -e "${GREEN}✓ Jupyter kernel installed${NC}\n"

# Check GPU availability (optional)
echo -e "${BLUE}Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    echo -e "${GREEN}✓ Found $GPU_COUNT GPU(s)${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠ No GPU detected (nvidia-smi not found)${NC}"
    echo -e "  Training will be slow without GPU!"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Development environment ready!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "Next steps:"
echo -e "  1. ${BLUE}Start Jupyter:${NC}  make jupyter"
echo -e "  2. ${BLUE}Open a notebook${NC} in your browser (http://localhost:8888)"
echo -e "  3. ${BLUE}Select kernel:${NC}  'LLM Finetuning' in Jupyter"
echo ""
echo -e "Useful commands:"
echo -e "  ${BLUE}make help${NC}      - Show all available commands"
echo -e "  ${BLUE}make format${NC}    - Format code with black & isort"
echo -e "  ${BLUE}make clean${NC}     - Clean up temporary files"
echo ""
echo -e "Documentation:"
echo -e "  ${BLUE}SETUP.md${NC}           - Setup guide"
echo -e "  ${BLUE}CONTRIBUTING.md${NC}    - How to contribute"
echo -e "  ${BLUE}ARCHITECTURE.md${NC}    - Design decisions"
echo ""
echo -e "Happy fine-tuning! 🚀"

