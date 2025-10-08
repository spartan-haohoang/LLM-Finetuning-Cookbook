# 📚 Full Fine-Tuning (From Scratch)

Full fine-tuning involves training all parameters of a language model, either from random initialization or from pre-trained weights. This is the most comprehensive but also the most resource-intensive approach.

---

## 📖 Overview

**When to use Full Fine-Tuning:**
- ✅ You have a domain-specific corpus (medical, legal, etc.)
- ✅ You need maximum model adaptation
- ✅ You have sufficient compute resources (multiple GPUs)
- ✅ Pre-trained models don't capture your domain well

**When NOT to use:**
- ❌ Limited compute resources
- ❌ Small dataset (< 1M tokens)
- ❌ Just need task-specific adaptation (use PEFT instead)

---

## 📓 Notebooks in This Section

### GPT-2-From-Scratch.ipynb

**Description**: Train a GPT-2 model from scratch on the `openwebtext` dataset.

**Key Concepts:**
- Model architecture configuration
- Large-scale dataset streaming with DeepLake
- Training loop optimization
- Memory management techniques

**Dependencies:**
```bash
# Install dependencies for this section
make install-full-finetuning

# Or manually:
poetry install --with full-finetuning
```

**Requirements:**
- **GPU**: NVIDIA GPU with 24GB+ VRAM (A100/V100 recommended)
- **Time**: 48-72 hours for full training
- **Storage**: ~50GB for dataset and checkpoints

**Quick Start:**
```bash
# Start Jupyter
poetry run jupyter lab

# Open the notebook
# Navigate to: 01-Full-Fine-Tuning/GPT-2-From-Scratch.ipynb
```

---

## 🔑 Key Techniques Covered

### 1. Model Configuration

```python
from transformers import GPT2Config

config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
)
```

### 2. Dataset Streaming

For large datasets that don't fit in memory:

```python
from datasets import load_dataset

dataset = load_dataset("openwebtext", streaming=True)
```

### 3. Mixed Precision Training

Reduce memory usage and speed up training:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    fp16=True,  # Mixed precision
    gradient_checkpointing=True,  # Save memory
)
```

### 4. Distributed Training

For multi-GPU setups:

```bash
# Using DeepSpeed
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json

# Using PyTorch DDP
torchrun --nproc_per_node=4 train.py
```

---

## 💡 Best Practices

### Memory Optimization

1. **Gradient Accumulation**: Simulate larger batches
   ```python
   training_args = TrainingArguments(
       per_device_train_batch_size=4,
       gradient_accumulation_steps=8,  # Effective batch size: 32
   )
   ```

2. **Gradient Checkpointing**: Trade compute for memory
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **8-bit Optimizers**: Reduce optimizer memory
   ```python
   training_args = TrainingArguments(
       optim="adamw_bnb_8bit",
   )
   ```

### Training Stability

1. **Gradient Clipping**: Prevent exploding gradients
   ```python
   training_args = TrainingArguments(
       max_grad_norm=1.0,
   )
   ```

2. **Warmup**: Stabilize early training
   ```python
   training_args = TrainingArguments(
       warmup_steps=1000,
   )
   ```

3. **Learning Rate Scheduling**: Improve convergence
   ```python
   training_args = TrainingArguments(
       lr_scheduler_type="cosine",
   )
   ```

---

## 📊 Expected Results

| Model | Dataset | Training Time | Final Loss | Perplexity |
|-------|---------|---------------|------------|------------|
| GPT-2 Small (124M) | OpenWebText (10%) | ~12 hours (4xV100) | ~3.2 | ~24 |
| GPT-2 Medium (355M) | OpenWebText (50%) | ~48 hours (8xA100) | ~2.8 | ~16 |
| GPT-2 Large (774M) | OpenWebText (100%) | ~120 hours (16xA100) | ~2.5 | ~12 |

*Note: Results vary based on hyperparameters and hardware.*

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**:
```python
# Reduce batch size
per_device_train_batch_size=1

# Enable gradient checkpointing
gradient_checkpointing=True

# Use smaller model
n_layer=6  # Instead of 12
```

### Issue: Training is Too Slow

**Solution**:
```python
# Enable mixed precision
fp16=True

# Increase batch size (if memory allows)
per_device_train_batch_size=8

# Use faster optimizer
optim="adamw_torch_fused"
```

### Issue: Loss Not Decreasing

**Solution**:
```python
# Check learning rate
learning_rate=5e-5  # Try different values

# Increase warmup
warmup_steps=2000

# Check data quality
# Ensure tokenization is correct
```

---

## 📚 Additional Resources

### Papers
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Improving Language Understanding by Generative Pre-Training (GPT)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### Tutorials
- [Hugging Face Training Guide](https://huggingface.co/docs/transformers/training)
- [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

### Tools
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Distributed training
- [DeepLake](https://github.com/activeloopai/deeplake) - Dataset streaming
- [Weights & Biases](https://wandb.ai/) - Experiment tracking

---

## 🎯 Next Steps

After mastering full fine-tuning, consider:

1. **[Parameter-Efficient Fine-Tuning (PEFT)](../02-PEFT/)** - Train with fewer resources
2. **[Instruction Tuning](../03-Instruction-Tuning/)** - Teach models to follow instructions
3. **Advanced Techniques**: RLHF, Constitutional AI, Multi-task Learning

---

## 💬 Need Help?

- Open an issue on GitHub
- Check the [SETUP.md](../SETUP.md) guide
- Review the [Troubleshooting section](#-troubleshooting)

