#!/bin/bash

# Quick Test Script
# Tests that the environment is set up correctly

set -e

echo "🧪 Testing LLM Finetuning Cookbook Setup"
echo "========================================"
echo ""

# Test 1: Poetry installation
echo "Test 1: Poetry..."
if command -v poetry &> /dev/null; then
    echo "✅ Poetry installed: $(poetry --version)"
else
    echo "❌ Poetry not found"
    exit 1
fi

# Test 2: Python version
echo ""
echo "Test 2: Python version..."
PYTHON_VERSION=$(poetry run python --version | awk '{print $2}')
echo "✅ Python version: $PYTHON_VERSION"

# Test 3: Core dependencies
echo ""
echo "Test 3: Core dependencies..."
poetry run python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
poetry run python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"
poetry run python -c "import datasets; print(f'✅ Datasets {datasets.__version__}')"
poetry run python -c "import jupyter; print(f'✅ Jupyter installed')"

# Test 4: GPU availability (if available)
echo ""
echo "Test 4: GPU availability..."
poetry run python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  No GPU detected (will use CPU)')
"

# Test 5: Optional dependencies (check if installed)
echo ""
echo "Test 5: Optional dependencies..."

# PEFT
if poetry run python -c "import peft" 2>/dev/null; then
    echo "✅ PEFT group installed"
else
    echo "ℹ️  PEFT group not installed (install with: make install-peft)"
fi

# Instruction Tuning
if poetry run python -c "import evaluate" 2>/dev/null; then
    echo "✅ Instruction Tuning group installed"
else
    echo "ℹ️  Instruction Tuning group not installed (install with: make install-instruction-tuning)"
fi

# Reasoning
if poetry run python -c "import trl" 2>/dev/null; then
    echo "✅ Reasoning group installed"
else
    echo "ℹ️  Reasoning group not installed (install with: make install-reasoning)"
fi

# Test 6: Jupyter kernel
echo ""
echo "Test 6: Jupyter kernel..."
if jupyter kernelspec list | grep -q "llm-finetuning"; then
    echo "✅ Jupyter kernel registered"
else
    echo "ℹ️  Jupyter kernel not registered (run: poetry run python -m ipykernel install --user --name llm-finetuning)"
fi

# Summary
echo ""
echo "========================================"
echo "✅ All tests passed!"
echo ""
echo "You're ready to start fine-tuning LLMs!"
echo "Run: make jupyter"
echo "========================================"

