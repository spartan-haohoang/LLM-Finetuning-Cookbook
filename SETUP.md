# 🚀 Setup Guide

This guide will help you set up your environment to run the LLM Finetuning Cookbook notebooks.

## 📋 Table of Contents

1. [Quick Start Options](#quick-start-options)
2. [Option 1: Local Setup with Poetry](#option-1-local-setup-with-poetry-recommended)
3. [Option 2: Docker Setup](#option-2-docker-setup)
4. [Option 3: VS Code Dev Containers](#option-3-vs-code-dev-containers)
5. [Dependency Groups Explained](#dependency-groups-explained)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start Options

Choose the setup method that best fits your workflow:

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Poetry (Local)** | Developers who want fine-grained control | Fast, flexible, minimal overhead | Requires manual Python/CUDA setup |
| **Docker** | Users who want zero-config setup | Isolated, reproducible | Larger download, slightly slower |
| **Dev Containers** | VS Code users | Best IDE integration | Requires VS Code + Docker |

---

## Option 1: Local Setup with Poetry (Recommended)

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** (recommended for training)
- **CUDA 12.1+** (if using GPU)
- **Poetry** (we'll install this)

### Step 1: Install Poetry

```bash
# Linux, macOS, Windows (WSL)
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
poetry --version
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/your-username/LLM-Finetuning-Cookbook.git
cd LLM-Finetuning-Cookbook
```

### Step 3: Choose Your Installation Type

#### 🔷 Option A: Install Everything (Recommended for Exploration)

```bash
# Install all dependencies for all notebooks
make install-all

# Or manually:
poetry install --with full-finetuning,peft,instruction-tuning,reasoning,dev
```

#### 🔷 Option B: Install Only What You Need

```bash
# Core dependencies only (minimal)
make install

# Then add specific groups as needed:
make install-full-finetuning    # For 01-Full-Fine-Tuning notebooks
make install-peft               # For 02-PEFT notebooks
make install-instruction-tuning # For 03-Instruction-Tuning notebooks
make install-reasoning          # For 04-Reasoning-Tuning notebooks
```

### Step 4: Launch Jupyter

```bash
# Start Jupyter Lab
make jupyter

# Or manually:
poetry run jupyter lab
```

Navigate to `http://localhost:8888` in your browser.

### Step 5 (Optional): Set Up Development Tools

```bash
# Install pre-commit hooks for code quality
make setup-dev
```

---

## Option 2: Docker Setup

Perfect for users who want a zero-configuration setup with GPU support.

### Prerequisites

- **Docker** (with BuildKit support)
- **Docker Compose**
- **NVIDIA Docker** (for GPU support)

### Step 1: Install Docker & NVIDIA Docker

```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Step 2: Build and Run

```bash
# Clone the repository
git clone https://github.com/your-username/LLM-Finetuning-Cookbook.git
cd LLM-Finetuning-Cookbook

# Build the Docker image
make docker-build

# Start Jupyter Lab
make docker-up
```

Navigate to `http://localhost:8888` in your browser.

### Docker Commands

```bash
# View logs
make docker-logs

# Stop containers
make docker-down

# Open a shell in the container
make docker-shell
```

---

## Option 3: VS Code Dev Containers

Best for VS Code users who want seamless IDE integration.

### Prerequisites

- **VS Code**
- **Docker**
- **Dev Containers extension** (install from VS Code marketplace)

### Setup

1. Install the "Dev Containers" extension in VS Code
2. Clone the repository
3. Open the repository in VS Code
4. Press `F1` → "Dev Containers: Reopen in Container"
5. Wait for the container to build (first time only)
6. Start working! Jupyter is already configured.

---

## Dependency Groups Explained

Our setup uses Poetry's **dependency groups** to organize requirements:

| Group | Description | Install Command |
|-------|-------------|-----------------|
| **core** (default) | Essential dependencies: `torch`, `transformers`, `datasets`, `jupyter` | `poetry install` |
| **full-finetuning** | For training from scratch: `deeplake`, `tensorboard` | `--with full-finetuning` |
| **peft** | For parameter-efficient fine-tuning: `peft`, `bitsandbytes` | `--with peft` |
| **instruction-tuning** | For instruction tuning: `evaluate`, `rouge-score`, `nltk` | `--with instruction-tuning` |
| **reasoning** | For reasoning tasks: `trl`, `einops` | `--with reasoning` |
| **dev** | Development tools: `black`, `pytest`, `pre-commit` | `--with dev` |

### Example: Running a Specific Notebook

**Scenario**: You want to run the Falcon-7B LoRA notebook (02-PEFT).

```bash
# Install core + PEFT dependencies
poetry install --with peft

# Start Jupyter
poetry run jupyter lab

# Open: 02-PEFT/Falcon-7B-LoRA.ipynb
```

---

## Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**

```python
# In your notebook, reduce batch size or use gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Reduce this
    gradient_accumulation_steps=8,   # Increase this
)
```

#### 2. **Poetry Not Found After Installation**

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc  # or source ~/.zshrc
```

#### 3. **NVIDIA Docker GPU Not Detected**

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, restart Docker
sudo systemctl restart docker
```

#### 4. **Port 8888 Already in Use**

```bash
# Use a different port
poetry run jupyter lab --port=8889
```

#### 5. **Slow Dependency Installation**

```bash
# Use a faster mirror (optional)
poetry config repositories.pypi https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 6. **Module Not Found in Jupyter**

Make sure Jupyter is using the Poetry environment:

```bash
# Install the kernel
poetry run python -m ipykernel install --user --name llm-finetuning

# Select this kernel in Jupyter: Kernel → Change Kernel → llm-finetuning
```

---

## 🎯 Next Steps

Once set up, check out:

1. **[README.md](README.md)** - Overview of all recipes
2. **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
3. **[docs/](docs/)** - Detailed guides for each technique

---

## 💡 Tips

- **Save Models**: Use `output_dir` in `TrainingArguments` to save checkpoints
- **Monitor Training**: Install `tensorboard` and view logs: `tensorboard --logdir=./runs`
- **Track Experiments**: Consider using [Weights & Biases](https://wandb.ai/) for experiment tracking
- **Memory Issues**: Use mixed precision (`fp16=True`) and gradient checkpointing

---

## 📞 Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Join our GitHub Discussions
- **Documentation**: Check out the [Hugging Face Transformers docs](https://huggingface.co/docs/transformers)

Happy fine-tuning! 🚀

