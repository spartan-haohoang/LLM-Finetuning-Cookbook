# 🏗️ Repository Architecture

This document explains the design decisions, patterns, and architecture of the LLM Finetuning Cookbook.

---

## 📐 Design Philosophy

### Core Principles

1. **User-Centric**: Zero configuration required for users
2. **Modular**: Install only what you need
3. **Professional**: Industry-standard tools and practices
4. **Educational**: Clear documentation and examples
5. **Maintainable**: Easy to extend and update

---

## 🎯 Dependency Management Strategy

### The Problem

Traditional approaches for multi-notebook repositories:
- ❌ Single `requirements.txt` → Bloated, conflicts, slow installs
- ❌ Per-notebook `requirements.txt` → Duplication, hard to maintain
- ❌ Per-notebook Dockerfile → Too many images, storage intensive

### Our Solution: Poetry with Dependency Groups

We use **Poetry** with **optional dependency groups** to achieve:

✅ **Single Source of Truth** (`pyproject.toml`)
✅ **Modular Installation** (install only what's needed)
✅ **Reproducibility** (locked versions in `poetry.lock`)
✅ **Professional** (modern Python standard)

#### Structure

```toml
[tool.poetry.dependencies]
python = "^3.10"
# Core dependencies (always installed)
torch = "^2.1.0"
transformers = "^4.35.0"
jupyter = "^1.0.0"

[tool.poetry.group.peft]
optional = true  # ← Key: Optional groups
# Only installed when: poetry install --with peft
peft = "^0.7.0"
bitsandbytes = "^0.41.0"

[tool.poetry.group.full-finetuning]
optional = true
deeplake = "^3.8.0"
# ... etc
```

#### Advantages Over Alternatives

| Aspect | Our Approach | Per-Notebook Deps | Single Requirements |
|--------|-------------|-------------------|---------------------|
| **Maintenance** | Single file | Multiple files | Single file |
| **Flexibility** | High (groups) | High (separate) | Low (all or nothing) |
| **Install Speed** | Fast (selective) | Medium | Slow (everything) |
| **Conflicts** | Resolved once | Per notebook | Resolved once |
| **Updates** | Centralized | Scattered | Centralized |
| **CI/CD** | Easy | Complex | Easy but slow |

---

## 🐳 Containerization Strategy

### Multi-Stage Dockerfile

We use a **multi-stage build** to support multiple use cases:

```dockerfile
# Stage 1: Base (CUDA, Python, Poetry)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 as base
# ... system dependencies, Poetry installation

# Stage 2: Development (all dependencies)
FROM base as dev
RUN poetry install --with full-finetuning,peft,instruction-tuning,reasoning,dev
# ... includes dev tools, all optional groups

# Stage 3: Production (minimal)
FROM base as prod
RUN poetry install --no-dev --no-root
# ... only core dependencies
```

**Benefits:**
- 🎯 **Flexible**: Users choose dev or prod
- 💾 **Efficient**: Layers are cached and reused
- 🔧 **Maintainable**: Single Dockerfile for all scenarios
- 🚀 **Fast**: Incremental builds

### Docker Compose Orchestration

```yaml
services:
  jupyter-all:
    build:
      target: dev  # Use dev stage
    ports: ["8888:8888"]
    volumes:
      - .:/workspace  # Live code mounting
      - huggingface-cache:/root/.cache/huggingface  # Persist models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
```

**Key Features:**
- GPU support via NVIDIA Docker
- Volume mounting for live development
- Persistent caching for models
- Easy orchestration with `docker-compose up`

### Why Not Per-Notebook Dockerfiles?

**Considered but rejected** for these reasons:

| Per-Notebook Dockerfiles | Our Approach |
|-------------------------|--------------|
| 5 different images to build | 1 image to build |
| ~100GB total storage | ~20GB total storage |
| Duplicate base layers | Shared base layers |
| 5 images to maintain | 1 Dockerfile to maintain |
| Harder to test combinations | Easy to test all notebooks |

---

## 💻 VS Code Dev Containers

### Why Dev Containers?

For VS Code users, Dev Containers provide:
- 🎯 **Zero Config**: Open and start coding immediately
- 🔧 **Extensions**: Pre-installed Python, Jupyter, etc.
- 🐳 **Consistent**: Same environment for all developers
- 🚀 **Fast**: Uses local Docker, no remote setup

### Configuration

```json
{
  "name": "LLM Finetuning Cookbook",
  "dockerComposeFile": ["../docker-compose.yml"],
  "service": "jupyter-all",
  "workspaceFolder": "/workspace",
  
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter",
        // ... more extensions
      ]
    }
  }
}
```

---

## 🛠️ Makefile: Developer Experience

### Philosophy

Developers shouldn't need to remember complex commands. The Makefile provides:

- 🎯 **Discoverability**: `make help` shows all commands
- 🚀 **Convenience**: `make jupyter` vs `poetry run jupyter lab --no-browser`
- 📚 **Documentation**: Each target has a description
- 🔄 **Consistency**: Same commands across environments

### Example

```makefile
install-peft: ## Install dependencies for PEFT notebooks
	@echo "$(BLUE)Installing PEFT dependencies...$(NC)"
	poetry install --with peft
```

Users run:
```bash
make install-peft  # Simple, memorable
```

Instead of:
```bash
poetry install --with peft --no-dev --sync  # Complex, easy to forget
```

---

## 📁 Directory Structure

### Flat vs. Nested

**Choice: Flat structure with numbered directories**

```
01-Full-Fine-Tuning/
02-PEFT/
03-Instruction-Tuning/
04-Reasoning-Tuning/
```

**Rationale:**
- ✅ Clear progression (beginners follow 01 → 04)
- ✅ Easy navigation
- ✅ Avoids deep nesting
- ✅ Works well with Jupyter
- ✅ Clean URLs when hosted

**Alternative considered:**
```
src/
  tutorials/
    beginner/
      full-finetuning/
    intermediate/
      peft/
```

**Rejected because:**
- ❌ More complex
- ❌ Harder to navigate
- ❌ Less clear for beginners
- ❌ Doesn't match educational flow

---

## 📝 Documentation Strategy

### Multi-Level Documentation

```
README.md              ← Overview, quick start
├── SETUP.md          ← Detailed installation guide
├── CONTRIBUTING.md   ← How to contribute
├── ARCHITECTURE.md   ← This file (design decisions)
│
├── 01-Full-Fine-Tuning/
│   └── README.md     ← Technique-specific guide
│
├── 02-PEFT/
│   └── README.md     ← Best practices, troubleshooting
│
└── ... (each directory has detailed README)
```

**Why this structure?**
- 🎯 **Progressive disclosure**: Start simple, go deeper as needed
- 🔍 **Findable**: Each level has its own documentation
- 📚 **Complete**: From quick start to deep dives
- 🔄 **DRY**: No duplication; cross-reference instead

---

## 🧪 Code Quality

### Pre-Commit Hooks

```yaml
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black  # Auto-format Python code
  
  - repo: https://github.com/kynan/nbstripout
    hooks:
      - id: nbstripout  # Clean notebook outputs
```

**Rationale:**
- ✅ Enforce consistency automatically
- ✅ Prevent common mistakes (committing outputs)
- ✅ Reduce code review overhead
- ✅ Professional standard

### Tools Choice

| Tool | Purpose | Why This One? |
|------|---------|---------------|
| **black** | Formatting | Industry standard, opinionated |
| **isort** | Import sorting | Works with black, configurable |
| **flake8** | Linting | Lightweight, extensible |
| **mypy** | Type checking | Optional, gradual typing |
| **nbstripout** | Notebook cleaning | Essential for notebooks |

---

## 🔄 CI/CD Considerations

### Design for CI/CD

Our structure makes CI/CD easy:

```yaml
# Example GitHub Actions workflow
- name: Install dependencies
  run: |
    poetry install --with peft
    
- name: Test notebook
  run: |
    poetry run jupyter nbconvert --to notebook --execute \
      02-PEFT/Falcon-7B-LoRA.ipynb
```

**Benefits:**
- ✅ Selective testing (test changed notebooks only)
- ✅ Fast (install only needed dependencies)
- ✅ Parallel execution (different groups on different runners)
- ✅ Matrix testing (test different Python versions)

---

## 🎓 Educational Design

### Notebook Structure

Each notebook follows this pattern:

```
1. Title + Description
   ↓
2. Setup + Dependencies
   ↓
3. Imports
   ↓
4. Data Loading
   ↓
5. Model Configuration
   ↓
6. Training
   ↓
7. Evaluation
   ↓
8. Inference Examples
   ↓
9. Next Steps
```

**Rationale:**
- 🎯 **Predictable**: Users know what to expect
- 📚 **Progressive**: Each step builds on previous
- 🔄 **Reusable**: Clear sections to copy/adapt
- 🧠 **Educational**: Natural learning flow

---

## 🚀 Performance Considerations

### Model Caching

```yaml
volumes:
  - huggingface-cache:/root/.cache/huggingface
```

**Benefit:** Models downloaded once, reused across containers

### Layer Caching

```dockerfile
# Copy only dependency files first
COPY pyproject.toml poetry.lock* ./
RUN poetry install ...

# Then copy code (changes frequently)
COPY . .
```

**Benefit:** Dependency layer cached, only code layer rebuilds

### Memory Efficiency

- Use 4-bit quantization where possible
- Provide memory-optimized variants
- Document GPU requirements clearly

---

## 🔮 Future-Proofing

### Extensibility Points

1. **Adding New Notebooks**:
   - Add to existing directory OR create new numbered directory
   - Add dependency group if needed
   - Update section README

2. **Adding New Techniques**:
   - Create new dependency group
   - Add Makefile target
   - Update main README

3. **Supporting New Models**:
   - Usually no structure changes needed
   - May need new dependency group for specialized libraries

### Migration Path

If Poetry becomes obsolete:
- `pyproject.toml` is PEP 621 standard
- Can migrate to `pip` + `pyproject.toml` (PEP 517)
- Dependency groups → extras_require
- Minimal disruption

---

## 📊 Comparison with Alternatives

### Alternative 1: Monolithic Approach

**Structure:**
```
requirements.txt       # Everything
notebooks/
  notebook1.ipynb
  notebook2.ipynb
```

**Pros:** Simple
**Cons:** Slow installs, conflicts, not modular

### Alternative 2: Per-Notebook Isolation

**Structure:**
```
notebook1/
  notebook.ipynb
  requirements.txt
  Dockerfile
notebook2/
  notebook.ipynb
  requirements.txt
  Dockerfile
```

**Pros:** Complete isolation
**Cons:** Duplication, hard to maintain, storage intensive

### Our Approach: Hybrid

**Structure:** Shared core + optional groups
**Pros:** Best of both worlds
**Cons:** Requires Poetry (acceptable trade-off)

---

## 🎯 Key Takeaways

1. **Poetry + Dependency Groups** = Modular, maintainable dependencies
2. **Multi-Stage Dockerfile** = Flexible containerization
3. **Makefile** = Great developer experience
4. **Comprehensive Docs** = Users can self-serve
5. **Pre-Commit Hooks** = Automated quality

This architecture balances:
- 👤 **User needs** (easy setup, flexibility)
- 🔧 **Maintainer needs** (easy updates, clear structure)
- 📚 **Educational goals** (clear progression, good examples)
- 💼 **Professional standards** (modern tools, best practices)

---

## 📚 Further Reading

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Dev Containers Spec](https://containers.dev/)
- [Makefile Tutorial](https://makefiletutorial.com/)

