# ⚡ Quick Start Guide

Get up and running with LLM Finetuning Cookbook in under 5 minutes!

---

## 🎯 Choose Your Path

### Path 1: I Want to Try Everything (Recommended for Learning)

```bash
# 1. Clone
git clone https://github.com/your-username/LLM-Finetuning-Cookbook.git
cd LLM-Finetuning-Cookbook

# 2. Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 3. Install everything
make install-all

# 4. Start Jupyter
make jupyter

# 5. Open any notebook and run!
# Navigate to: http://localhost:8888
```

**Time:** ~10 minutes (depending on download speed)
**Disk Space:** ~15GB
**Best For:** Learners, explorers, contributors

---

### Path 2: I Want a Specific Notebook Only

```bash
# 1. Clone
git clone https://github.com/your-username/LLM-Finetuning-Cookbook.git
cd LLM-Finetuning-Cookbook

# 2. Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 3. Install core + specific dependencies
# Choose ONE:

make install-peft               # For LoRA/QLoRA (Falcon-7B)
make install-instruction-tuning # For summarization/sentiment
make install-reasoning          # For math reasoning
make install-full-finetuning    # For GPT-2 from scratch

# 4. Start Jupyter
make jupyter
```

**Time:** ~5 minutes
**Disk Space:** ~5-8GB (per section)
**Best For:** Focused learning, production use

---

### Path 3: I Want Zero Configuration (Docker)

```bash
# 1. Clone
git clone https://github.com/your-username/LLM-Finetuning-Cookbook.git
cd LLM-Finetuning-Cookbook

# 2. Start with Docker
make docker-build
make docker-up

# 3. Open browser
# Navigate to: http://localhost:8888
```

**Time:** ~15 minutes (first time, includes Docker image build)
**Disk Space:** ~20GB
**Best For:** Users who love Docker, consistent environments

---

### Path 4: I Use VS Code

1. **Install** Docker + "Dev Containers" extension
2. **Clone** the repository
3. **Open** in VS Code
4. **Press** `F1` → "Dev Containers: Reopen in Container"
5. **Done!** Everything configured automatically

**Time:** ~10 minutes (first time)
**Best For:** VS Code users, team environments

---

## 🎓 Your First Notebook

### Beginner: Start with PEFT (Easiest)

```bash
# Install
make install-peft

# Open
poetry run jupyter lab

# Navigate to: 02-PEFT/Falcon-7B-LoRA.ipynb
# Run all cells (Shift+Enter)
```

**Why this one?**
- ✅ Fastest to run (2-3 hours)
- ✅ Lowest memory (16GB GPU)
- ✅ Most practical (widely used technique)
- ✅ Great results with minimal resources

---

## 🚀 Quick Command Reference

```bash
# Installation
make install              # Core only
make install-all          # Everything
make install-peft         # PEFT only
make install-reasoning    # Reasoning only

# Running
make jupyter              # Start Jupyter Lab
make docker-up            # Start Docker container

# Maintenance
make clean                # Clean up temp files
make format               # Format code
make update               # Update dependencies

# Help
make help                 # Show all commands
```

---

## 🎯 What to Expect

### PEFT Notebooks (Recommended Start)
- **Time:** 2-4 hours
- **GPU:** 16GB+ (RTX 4090, A100)
- **Output:** Fine-tuned model adapters (~100MB)

### Instruction Tuning Notebooks
- **Time:** 1-3 hours
- **GPU:** 8GB+ (RTX 3070+)
- **Output:** Task-specific models

### Reasoning Notebooks (Advanced)
- **Time:** 6-12 hours
- **GPU:** 24GB+ (A100/A6000)
- **Output:** Math-capable models

### Full Fine-Tuning (Most Advanced)
- **Time:** 24-72 hours
- **GPU:** 80GB+ or multi-GPU
- **Output:** Complete pre-trained model

---

## 🐛 Common Issues & Quick Fixes

### Issue: "Poetry not found"

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

### Issue: "CUDA out of memory"

```python
# In the notebook, find TrainingArguments and reduce:
per_device_train_batch_size=1  # Reduce this
gradient_accumulation_steps=8   # Increase this
```

### Issue: "Port 8888 already in use"

```bash
# Use a different port
poetry run jupyter lab --port=8889
```

### Issue: Docker not detecting GPU

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## 📚 Next Steps

### After Running Your First Notebook:

1. **Read the Section README**
   - Each directory has a detailed README
   - Learn best practices and techniques

2. **Experiment**
   - Change hyperparameters
   - Try different models
   - Use your own data

3. **Explore Other Sections**
   - Progress from PEFT → Instruction Tuning → Reasoning
   - Build on what you learned

4. **Contribute**
   - Found a bug? Open an issue
   - Made improvements? Submit a PR
   - See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🎓 Learning Path

```
Beginner
  └─ 02-PEFT (LoRA/QLoRA)              [Start Here!]
       │
       ├─ 03-Instruction-Tuning         [Next: Practical Tasks]
       │    │
       │    └─ Summarization
       │    └─ Sentiment Analysis
       │
       ├─ 04-Reasoning-Tuning           [Advanced: Math & Logic]
       │
       └─ 01-Full-Fine-Tuning           [Expert: From Scratch]
```

**Recommendation:** Start with **02-PEFT**, then explore based on your interests!

---

## 💡 Pro Tips

1. **Start Small**: Use small datasets first to test your setup
2. **Monitor GPU**: Use `nvidia-smi` to watch memory usage
3. **Save Checkpoints**: Training can be interrupted, save often!
4. **Use Logs**: Enable logging to track progress
5. **Read Docs**: Each section's README has valuable insights

---

## 🆘 Need Help?

- 📖 **Detailed Setup**: See [SETUP.md](SETUP.md)
- 🐛 **Troubleshooting**: Check section READMEs
- 💬 **Questions**: Open a GitHub issue
- 🤝 **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## ⏱️ Time Investment Summary

| What You Want | Time to Start | Time to Complete |
|---------------|---------------|------------------|
| Explore codebase | 5 min | N/A |
| Run first notebook | 15 min | 2-4 hours |
| Learn PEFT | 1 hour | 1 day |
| Master all techniques | 1 day | 1 week |

---

**Ready?** Pick a path above and start your LLM fine-tuning journey! 🚀

