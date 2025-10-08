# LLM Finetuning Cookbook

🔥 A hands-on, code-first collection of recipes for fine-tuning Large Language Models. 🔥

This repository is not a polished library but a collection of practical, well-commented notebooks and scripts. The goal is to provide deep insights into the code and mechanics behind various LLM fine-tuning techniques

---

## 📖 Table of Contents

This cookbook is divided into several key fine-tuning methodologies. Each section contains dedicated code, explanations, and links to relevant resources.

1.  [**📚 Full Fine-Tuning (From Scratch)**](./01-Full-Fine-Tuning/): Modifying every weight in the model. The most thorough but resource-intensive method.
2.  [**⚡ Parameter-Efficient Fine-Tuning (PEFT)**](./02-PEFT/): Smart techniques to fine-tune LLMs with a fraction of the computational cost.
3.  [**🎯 Instruction & Task Fine-Tuning**](./03-Instruction-Tuning/): Teaching a model to follow instructions and perform specific tasks like summarization or sentiment analysis.
4.  [**🧠 Reasoning Fine-Tuning**](./04-Reasoning-Tuning/): Enhancing a model's ability to perform logical, mathematical, or multi-step reasoning.

---

### **1. 📚 Full Fine-Tuning (From Scratch)**

In this section, we explore the foundational approach of training all of the model's parameters on a new dataset. This is ideal for adapting a model to a completely new domain or style.

**Featured Recipe: Training GPT-2 From Scratch**
* **Description**: A step-by-step guide to pre-training a GPT-2 model from the ground up on the `openwebtext` dataset.
* **Key Concepts Covered**:
    * Loading and streaming large datasets with `Deep Lake`.
    * Configuring model architecture (`n_layer`, `n_head`, `n_embd`).
    * Setting up the `Hugging Face Trainer` and `TrainingArguments`.
    * Running the training loop and performing inference.
* **Code**: [`./01-Full-Fine-Tuning/GPT-2-From-Scratch.ipynb`](./01-Full-Fine-Tuning/GPT-2-From-Scratch.ipynb)

---

### **2. ⚡ Parameter-Efficient Fine-Tuning (PEFT)**

PEFT methods allow us to achieve great performance by only training a small subset of the model's parameters. This makes fine-tuning accessible without high-end hardware.

**Featured Recipe: Fine-tuning Falcon-7B with LoRA**
* **Description**: Use Low-Rank Adaptation (LoRA) to efficiently fine-tune the powerful Falcon-7B model on a custom dataset.
* **Key Concepts Covered**:
    * **Quantization**: Loading models in 4-bit using `BitsAndBytesConfig` to save memory.
    * **LoRA Configuration**: Setting up `LoraConfig` from the `peft` library (`r`, `lora_alpha`, `target_modules`).
    * **SFTTrainer**: Using the Supervised Fine-Tuning trainer from the `trl` library for simplified training.
* **Code**: [`./02-PEFT/Falcon-7B-LoRA.ipynb`](./02-PEFT/Falcon-7B-LoRA.ipynb)

---

### **3. 🎯 Instruction & Task Fine-Tuning**

This is the most common type of fine-tuning, where we teach a base model to act as a helpful assistant for specific tasks by training it on instruction-response pairs.

**Featured Recipes:**
1.  **Summarization with FLAN-T5**:
    * **Description**: Fine-tune Google's FLAN-T5 on the `dialogsum` dataset to make it an expert summarizer. Compares full fine-tuning vs. PEFT (LoRA).
    * **Key Concepts**: Data preprocessing for instruction-following, evaluating with the ROUGE metric.
    * **Code**: [`./03-Instruction-Tuning/Summarization-FLAN-T5.ipynb`](./03-Instruction-Tuning/Summarization-FLAN-T5.ipynb)

2.  **Financial Sentiment Analysis with OPT-1.3b**:
    * **Description**: Adapt the OPT-1.3b model to understand financial news sentiment.
    * **Key Concepts**: Domain-specific adaptation, combining LoRA with instruction tuning for a specialized task.
    * **Code**: [`./03-Instruction-Tuning/Financial-Sentiment-OPT.ipynb`](./03-Instruction-Tuning/Financial-Sentiment-OPT.ipynb)

---

### **4. 🧠 Reasoning Fine-Tuning**

A more advanced topic focused on improving a model's ability to "think" through complex problems, from math puzzles to logical deductions.

**Featured Recipe: Mathematical Reasoning with Qwen & GRPO**
* **Description**: Enhance the mathematical capabilities of the Qwen model using advanced techniques like Generalized Reward Policy Optimization (GRPO).
* **Key Concepts Covered**:
    * Using specialized datasets for reasoning tasks.
    * Leveraging high-performance libraries like `Unsloth` for faster training.
    * Implementing advanced optimization techniques beyond standard fine-tuning.
* **Code**: [`./04-Reasoning-Tuning/Math-Reasoning-Qwen-GRPO.ipynb`](./04-Reasoning-Tuning/Math-Reasoning-Qwen-GRPO.ipynb)

---

## 📂 Project Structure

```
LLM-Finetuning-Cookbook/
│
├── 01-Full-Fine-Tuning/
│   ├── GPT-2-From-Scratch.ipynb
│   └── README.md                      # Detailed guide for this section
│
├── 02-PEFT/
│   ├── Falcon-7B-LoRA.ipynb
│   └── README.md                      # LoRA, QLoRA techniques explained
│
├── 03-Instruction-Tuning/
│   ├── Summarization-FLAN-T5.ipynb
│   ├── Financial-Sentiment-OPT.ipynb
│   └── README.md                      # Instruction tuning best practices
│
├── 04-Reasoning-Tuning/
│   ├── Math-Reasoning-Qwen-GRPO.ipynb
│   └── README.md                      # Advanced reasoning techniques
│
├── .devcontainer/
│   └── devcontainer.json              # VS Code Dev Container config
│
├── pyproject.toml                     # Poetry dependency management
├── Dockerfile                         # Multi-stage Docker build
├── docker-compose.yml                 # Container orchestration
├── Makefile                          # Convenient commands
├── .pre-commit-config.yaml           # Code quality hooks
├── .gitignore
├── .dockerignore
├── SETUP.md                          # 📖 Complete setup guide
├── CONTRIBUTING.md                   # 🤝 Contribution guidelines
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Quick Start (3 Options)

Choose your preferred setup method:

#### 🔷 Option 1: Local Setup with Poetry (Recommended)

```bash
# Clone repository
git clone https://github.com/your-username/LLM-Finetuning-Cookbook.git
cd LLM-Finetuning-Cookbook

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies
make install-all

# Start Jupyter Lab
make jupyter
```

#### 🐳 Option 2: Docker Setup (Zero Configuration)

```bash
# Clone and start with Docker
git clone https://github.com/your-username/LLM-Finetuning-Cookbook.git
cd LLM-Finetuning-Cookbook

# Build and run
make docker-build
make docker-up

# Open http://localhost:8888
```

#### 💻 Option 3: VS Code Dev Containers (Best IDE Integration)

1. Install Docker and the "Dev Containers" extension
2. Open repository in VS Code
3. Press `F1` → "Dev Containers: Reopen in Container"
4. Everything is configured automatically!

### Detailed Setup Instructions

For comprehensive setup guides, troubleshooting, and advanced configurations, see:

📖 **[SETUP.md](SETUP.md)** - Complete installation guide for all methods

### Prerequisites

* **Python 3.10+**
* **Poetry** (for local setup)
* **Docker** (for containerized setup)
* **NVIDIA GPU** (highly recommended for training)
* **CUDA 12.1+** (if using GPU)

---

## 🎯 Selective Installation

Don't need everything? Install only what you need:

```bash
# Core dependencies only
make install

# Add specific sections as needed
make install-full-finetuning    # For 01-Full-Fine-Tuning
make install-peft               # For 02-PEFT  
make install-instruction-tuning # For 03-Instruction-Tuning
make install-reasoning          # For 04-Reasoning-Tuning

# Or combine multiple
poetry install --with peft,instruction-tuning
```

---

## 🛠️ Useful Commands

```bash
# Development
make jupyter           # Start Jupyter Lab
make format           # Format code with black & isort
make lint             # Run linters
make clean            # Clean up generated files

# Docker
make docker-build     # Build Docker image
make docker-up        # Start containers
make docker-down      # Stop containers
make docker-shell     # Open shell in container

# Maintenance
make lock             # Update poetry.lock
make update           # Update dependencies
```

---

## 🏗️ Repository Design Patterns

This repository follows modern best practices:

### ✅ Dependency Management
- **Poetry** with dependency groups for modular installations
- Single `pyproject.toml` as source of truth
- Locked dependencies for reproducibility

### ✅ Containerization
- Multi-stage Dockerfile (dev/prod)
- Docker Compose for orchestration
- VS Code Dev Containers for seamless development
- GPU support with NVIDIA Docker

### ✅ Code Quality
- Pre-commit hooks (black, isort, flake8)
- Automated formatting and linting
- Type hints where applicable
- Clean notebook outputs before commits

### ✅ Documentation
- Comprehensive README per section
- Detailed setup guide (SETUP.md)
- Contributing guidelines (CONTRIBUTING.md)
- Inline comments in all notebooks

### ✅ Professional Structure
- Consistent notebook formatting
- Modular dependencies
- Easy CI/CD integration
- Beginner to advanced friendly

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- How to add new notebooks
- Code quality standards
- Testing guidelines
- Pull request process

**Quick contribution setup:**
```bash
make setup-dev  # Install dev dependencies + pre-commit hooks
```

---

## 🙏 Acknowledgements

A huge thank you to **Youssef Hosni** for his clear, detailed, and practical articles that form the inspiration for this repository. Please check out his work:

* **GitHub**: [youssefHosni/Hands-On-LLM-Fine-Tuning](https://github.com/youssefHosni/Hands-On-LLM-Fine-Tuning)
* **Medium**: [Youssef Hosni's Articles](https://youssef-hosni.medium.com/)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
