# ⚡ Parameter-Efficient Fine-Tuning (PEFT)

PEFT methods enable fine-tuning large language models by training only a small subset of parameters, making it accessible even on consumer hardware.

---

## 📖 Overview

**Why PEFT?**
- 💰 **Cost-Effective**: Train on a single GPU instead of expensive clusters
- ⚡ **Fast**: Fewer parameters = faster training
- 💾 **Memory Efficient**: Load models in 4-bit/8-bit quantization
- 🎯 **Performance**: Often matches full fine-tuning quality
- 🔄 **Modularity**: Easily swap adapters for different tasks

**Popular PEFT Techniques:**
- **LoRA** (Low-Rank Adaptation) - Most widely used
- **QLoRA** - LoRA + 4-bit quantization
- **Prefix Tuning** - Learn prompt embeddings
- **Adapter Layers** - Insert trainable modules
- **IA³** (Infused Adapter) - Scale activations

---

## 📓 Notebooks in This Section

### Falcon-7B-LoRA.ipynb

**Description**: Fine-tune the powerful Falcon-7B model using LoRA on a custom dataset.

**Key Concepts:**
- 4-bit quantization with `bitsandbytes`
- LoRA configuration and target modules
- `SFTTrainer` from the `trl` library
- Merging and saving adapters

**Dependencies:**
```bash
# Install dependencies for this section
make install-peft

# Or manually:
poetry install --with peft
```

**Requirements:**
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100, etc.)
- **Time**: 2-4 hours for typical datasets
- **Storage**: ~20GB for model and adapters

**Quick Start:**
```bash
# Start Jupyter
poetry run jupyter lab

# Open the notebook
# Navigate to: 02-PEFT/Falcon-7B-LoRA.ipynb
```

---

## 🔑 Key Techniques

### 1. LoRA Configuration

LoRA works by injecting trainable low-rank matrices into the model:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                      # Rank of decomposition (4-64 typical)
    lora_alpha=32,             # Scaling factor (usually 2*r)
    target_modules=[           # Which layers to apply LoRA
        "query_key_value",     # Attention layers
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
    lora_dropout=0.05,         # Dropout for regularization
    bias="none",               # Don't train bias terms
    task_type="CAUSAL_LM",     # Task type
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4M (0.05%) || all params: 7B (100%)
```

### 2. 4-bit Quantization

Reduce memory by loading models in 4-bit precision:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,     # Nested quantization
    bnb_4bit_quant_type="nf4",          # NormalFloat4 (best for LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype
)

model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### 3. SFTTrainer (Supervised Fine-Tuning)

Simplified training with the `trl` library:

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
```

---

## 📊 Memory & Performance Comparison

| Method | Trainable Params | GPU Memory | Training Speed | Quality |
|--------|------------------|------------|----------------|---------|
| Full Fine-Tuning | 7B (100%) | 80GB+ | 1x | ⭐⭐⭐⭐⭐ |
| LoRA (16-bit) | 3.5M (0.05%) | 28GB | 1.5x | ⭐⭐⭐⭐ |
| QLoRA (4-bit) | 3.5M (0.05%) | 10GB | 1.2x | ⭐⭐⭐⭐ |

*Based on Falcon-7B with batch size 4 on A100 40GB.*

---

## 💡 Best Practices

### Choosing LoRA Hyperparameters

**Rank (`r`)**:
- **Small (4-8)**: Simple tasks, limited data
- **Medium (16-32)**: Most use cases (recommended)
- **Large (64-128)**: Complex tasks, lots of data

**Alpha (`lora_alpha`)**:
- Rule of thumb: `lora_alpha = 2 * r`
- Higher alpha = stronger LoRA effect

**Target Modules**:
```python
# Minimal (faster, less memory)
target_modules=["q_proj", "v_proj"]

# Balanced (recommended)
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# Comprehensive (best quality)
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Memory Optimization Tips

1. **Gradient Checkpointing**:
   ```python
   model.gradient_checkpointing_enable()
   ```

2. **Smaller Batch Size + Gradient Accumulation**:
   ```python
   per_device_train_batch_size=1
   gradient_accumulation_steps=16
   ```

3. **Paged Optimizers** (for limited VRAM):
   ```python
   optim="paged_adamw_8bit"
   ```

### Model-Specific Target Modules

Different models have different layer names:

```python
# Llama, Mistral, Mixtral
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Falcon
target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

# GPT-2, GPT-J
target_modules=["c_attn", "c_proj"]

# BLOOM
target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
```

---

## 🚀 Advanced Techniques

### QLoRA (Quantized LoRA)

The most memory-efficient approach:

```python
# Load in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Apply LoRA
lora_config = LoraConfig(r=64, lora_alpha=128, target_modules="all-linear")

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

### Merging Adapters

After training, merge LoRA weights back into base model:

```python
# Load base model and adapter
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b")
model = PeftModel.from_pretrained(model, "path/to/adapter")

# Merge
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("falcon-7b-finetuned")
```

### Multi-Adapter Inference

Load different adapters for different tasks:

```python
# Load base model
model = AutoModelForCausalLM.from_pretrained("falcon-7b")

# Load multiple adapters
model = PeftModel.from_pretrained(model, "adapter-summarization", adapter_name="summarize")
model.load_adapter("adapter-chat", adapter_name="chat")

# Switch between adapters
model.set_adapter("summarize")
output = model.generate(...)

model.set_adapter("chat")
output = model.generate(...)
```

---

## 📈 Evaluation & Monitoring

### Track Training Progress

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=500,
    load_best_model_at_end=True,
    report_to="tensorboard",  # or "wandb"
)
```

### Visualize with TensorBoard

```bash
tensorboard --logdir=./logs
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solutions**:
```python
# 1. Enable 4-bit quantization
load_in_4bit=True

# 2. Reduce batch size
per_device_train_batch_size=1

# 3. Enable gradient checkpointing
gradient_checkpointing=True

# 4. Reduce sequence length
max_seq_length=1024  # Instead of 2048
```

### Issue: Poor Quality Results

**Solutions**:
```python
# 1. Increase LoRA rank
r=32  # or 64

# 2. Target more modules
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 3. Train longer
num_train_epochs=5

# 4. Tune learning rate
learning_rate=2e-4  # Try different values
```

### Issue: Training is Slow

**Solutions**:
```python
# 1. Use bfloat16 (if supported)
bf16=True

# 2. Increase batch size
per_device_train_batch_size=4

# 3. Reduce target modules
target_modules=["q_proj", "v_proj"]  # Minimal set
```

---

## 📚 Additional Resources

### Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Prefix-Tuning](https://arxiv.org/abs/2101.00190)

### Documentation
- [PEFT Library](https://huggingface.co/docs/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)

### Tutorials
- [Hugging Face PEFT Guide](https://huggingface.co/blog/peft)
- [QLoRA Tutorial](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

---

## 🎯 Next Steps

- Try other PEFT methods: Prefix Tuning, Adapter Layers
- Explore [Instruction Tuning](../03-Instruction-Tuning/)
- Learn about [Reasoning Fine-Tuning](../04-Reasoning-Tuning/)

---

## 💬 Need Help?

- Check the [SETUP.md](../SETUP.md) guide
- Open an issue on GitHub
- Review [Troubleshooting](#-troubleshooting)

