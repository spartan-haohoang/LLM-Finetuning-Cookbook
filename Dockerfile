# Multi-stage Dockerfile for LLM Finetuning Cookbook
# This provides a containerized environment with all dependencies

# Base stage with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true \
    PATH="/opt/poetry/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Development stage (all dependencies)
FROM base as dev

# Install all dependencies including optional groups
RUN poetry install --with full-finetuning,peft,instruction-tuning,reasoning,dev --no-root

# Copy the rest of the application
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Set Jupyter to listen on all interfaces
CMD ["poetry", "run", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

# Production stage (minimal dependencies)
FROM base as prod

# Install only core dependencies
RUN poetry install --no-dev --no-root

# Copy only necessary files
COPY . .

EXPOSE 8888

CMD ["poetry", "run", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

