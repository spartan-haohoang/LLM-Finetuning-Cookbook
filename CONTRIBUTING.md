# 🤝 Contributing to LLM Finetuning Cookbook

Thank you for your interest in contributing! This guide will help you get started.

## 📋 Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest new fine-tuning recipes
- 📝 Improve documentation
- 🔧 Fix issues
- ✨ Add new notebooks

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/LLM-Finetuning-Cookbook.git
cd LLM-Finetuning-Cookbook
```

### 2. Set Up Development Environment

```bash
# Install all dependencies including dev tools
make setup-dev

# Or manually:
poetry install --with full-finetuning,peft,instruction-tuning,reasoning,dev
poetry run pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## 📝 Adding a New Notebook

### Structure

Each notebook should follow this structure:

```python
# Cell 1: Title and Description
"""
# [Technique Name] - [Model Name]

**Description**: Brief explanation of what this notebook does.

**Key Concepts**:
- Concept 1
- Concept 2
- Concept 3

**Dependencies**: List specific dependencies needed
"""

# Cell 2: Install Dependencies (if needed)
# !pip install specific-package==version

# Cell 3: Imports
import torch
from transformers import ...
# ... other imports

# Cell 4-N: Implementation with detailed comments
# ... your code with explanations
```

### Checklist for New Notebooks

- [ ] Clear markdown cells explaining each step
- [ ] Well-commented code
- [ ] Example output/results
- [ ] Works with specified dependencies
- [ ] No hardcoded paths (use `./data`, `./models`, etc.)
- [ ] GPU memory requirements documented
- [ ] Expected training time documented
- [ ] Links to relevant papers/resources

### Adding Dependencies

If your notebook needs new dependencies:

1. **Identify the appropriate group** (or create a new one)
2. **Add to `pyproject.toml`**:

```toml
[tool.poetry.group.your-new-group]
optional = true

[tool.poetry.group.your-new-group.dependencies]
new-package = "^1.0.0"
```

3. **Update the Makefile** with an install target:

```makefile
install-your-new-group: ## Install dependencies for Your New Group
	poetry install --with your-new-group
```

4. **Update SETUP.md** with the new group information

---

## 🧪 Code Quality Standards

### Python Code

We use:

- **Black** for formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking (where applicable)

```bash
# Format your code
make format

# Check linting
make lint
```

### Notebook Guidelines

- **Clean outputs** before committing (pre-commit hook does this automatically)
- **Keep cells small** and focused
- **Use markdown liberally** to explain concepts
- **Include visualizations** where helpful
- **Test on a small dataset** first (if applicable)

### Commit Messages

Follow conventional commits:

```
feat: add RLHF fine-tuning notebook
fix: correct learning rate in GPT-2 notebook
docs: update setup instructions for Windows
style: format code with black
refactor: reorganize PEFT examples
test: add validation for summarization
```

---

## 🔍 Testing Your Changes

### Test Locally

```bash
# Test notebook runs without errors
poetry run jupyter nbconvert --to notebook --execute your-notebook.ipynb

# Or open in Jupyter and run all cells
make jupyter
```

### Test Docker Build

```bash
# Ensure Docker image builds successfully
make docker-build

# Test running in Docker
make docker-up
# Check http://localhost:8888
```

---

## 📤 Submitting Changes

### 1. Ensure Quality

```bash
# Run all checks
make format
make lint
make clean
```

### 2. Commit and Push

```bash
git add .
git commit -m "feat: add your amazing contribution"
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. Fill in the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature (notebook)
- [ ] Documentation update
- [ ] Code refactoring

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Notebook runs without errors
- [ ] Dependencies added to pyproject.toml (if needed)

## Testing
Describe how you tested your changes

## Screenshots (if applicable)
Add any relevant screenshots or outputs
```

---

## 🎯 Contribution Ideas

### Easy (Good First Issues)

- Fix typos in documentation
- Add code comments to existing notebooks
- Improve error messages
- Add type hints

### Medium

- Add new fine-tuning technique
- Create visualization for training metrics
- Optimize existing notebooks for memory usage
- Add unit tests for utility functions

### Advanced

- Implement distributed training example
- Add multi-GPU support guide
- Create custom training loop examples
- Integrate with experiment tracking tools (W&B, MLflow)

---

## 📚 Resources

### Understanding the Codebase

- **pyproject.toml**: Dependency management with Poetry groups
- **Makefile**: Convenience commands for development
- **Dockerfile**: Multi-stage build for development/production
- **docker-compose.yml**: Orchestration for containers
- **.devcontainer/**: VS Code Dev Container configuration

### Learn More About Fine-Tuning

- [Hugging Face Course](https://huggingface.co/course)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [DeepSpeed](https://www.deepspeed.ai/)

---

## 🙏 Recognition

Contributors will be:

- Listed in README.md acknowledgements
- Credited in commit history
- Mentioned in release notes (for significant contributions)

---

## ❓ Questions?

- Open a GitHub Discussion
- Ask in the PR comments
- Open an issue with the `question` label

---

Thank you for making LLM Finetuning Cookbook better! 🚀

