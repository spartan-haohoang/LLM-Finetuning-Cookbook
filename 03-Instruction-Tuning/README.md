# 🎯 Instruction & Task Fine-Tuning

Instruction tuning teaches models to follow instructions and perform specific tasks, transforming base models into helpful assistants.

---

## 📖 Overview

**What is Instruction Tuning?**

Instruction tuning trains models on (instruction, response) pairs to make them better at:
- Following user instructions
- Performing specific tasks (summarization, Q&A, sentiment analysis)
- Generalizing to new tasks (zero-shot/few-shot learning)

**Key Benefits:**
- 🎯 **Task-Specific**: Excel at particular tasks
- 🧠 **Better Reasoning**: Understand intent and context
- 🔄 **Generalizable**: Transfer learning to similar tasks
- 👤 **User-Friendly**: Natural conversational interface

---

## 📓 Notebooks in This Section

### 1. Summarization-FLAN-T5.ipynb

**Description**: Fine-tune Google's FLAN-T5 on the DialogSum dataset to create an expert summarizer.

**Key Concepts:**
- Encoder-decoder architecture (T5)
- Dataset preprocessing for instruction-following
- ROUGE metric evaluation
- Comparing full fine-tuning vs. LoRA

**Use Case**: Summarizing conversations, documents, articles

**Dependencies:**
```bash
make install-instruction-tuning
```

**Requirements:**
- **GPU**: 8GB+ VRAM (RTX 3070 or better)
- **Time**: 1-2 hours
- **Dataset**: DialogSum (~13k dialogues)

---

### 2. Financial-Sentiment-OPT.ipynb

**Description**: Adapt Meta's OPT-1.3B to understand financial news sentiment.

**Key Concepts:**
- Domain-specific adaptation
- Classification task formatting
- LoRA for efficient training
- Evaluation on financial data

**Use Case**: Financial analysis, market sentiment tracking

**Dependencies:**
```bash
make install-instruction-tuning
poetry install --with peft  # Also needs PEFT
```

**Requirements:**
- **GPU**: 12GB+ VRAM (RTX 3090 or better)
- **Time**: 2-3 hours
- **Dataset**: Financial PhraseBank

---

## 🔑 Key Techniques

### 1. Prompt Formatting

Structure your data as instruction-response pairs:

```python
# Template
template = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""

# Example: Summarization
instruction = "Summarize the following conversation:"
input_text = "User: Hi, I need help with my order..."
response = "The user contacted support about a delayed order."

formatted = template.format(
    instruction=instruction,
    input=input_text,
    response=response
)
```

### 2. Dataset Preparation

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("knkarthick/dialogsum")

# Format function
def format_instruction(example):
    return {
        "text": f"""### Instruction:
Summarize the following dialogue:

### Input:
{example['dialogue']}

### Response:
{example['summary']}"""
    }

# Apply formatting
dataset = dataset.map(format_instruction)
```

### 3. Evaluation Metrics

#### ROUGE (for summarization)

```python
from datasets import load_metric

rouge = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    return {
        "rouge1": result["rouge1"].mid.fmeasure,
        "rouge2": result["rouge2"].mid.fmeasure,
        "rougeL": result["rougeL"].mid.fmeasure,
    }
```

#### Accuracy (for classification)

```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }
```

---

## 📊 Popular Datasets for Instruction Tuning

| Dataset | Size | Task | Best For |
|---------|------|------|----------|
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 52K | General instructions | Building assistants |
| [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) | 13K | Summarization | Conversation summaries |
| [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) | 4.8K | Sentiment | Financial analysis |
| [SQuAD](https://huggingface.co/datasets/squad) | 100K | Q&A | Question answering |
| [FLAN](https://github.com/google-research/FLAN) | 1.8M | Multi-task | General-purpose |
| [ShareGPT](https://huggingface.co/datasets/RyokoAI/ShareGPT52K) | 52K | Conversation | Chatbots |

---

## 💡 Best Practices

### Data Quality > Quantity

```python
# Good example
{
    "instruction": "Classify the sentiment of the following financial news.",
    "input": "Apple Inc. reported record quarterly revenue, beating analyst expectations.",
    "output": "Positive"
}

# Bad example (vague)
{
    "instruction": "Analyze this.",
    "input": "AAPL up",
    "output": "good"
}
```

### Balanced Training

- ✅ Mix different instruction types
- ✅ Include diverse examples
- ✅ Balance positive/negative/neutral examples
- ❌ Don't overtrain on a single pattern

### Hyperparameters for Instruction Tuning

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    learning_rate=2e-5,           # Lower than pre-training
    num_train_epochs=3,           # 2-5 epochs typical
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,             # 10% warmup
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    logging_steps=10,
    fp16=True,                    # Mixed precision
)
```

---

## 🚀 Advanced Techniques

### Multi-Task Instruction Tuning

Train on multiple tasks simultaneously:

```python
tasks = [
    {"task": "summarization", "weight": 0.4},
    {"task": "qa", "weight": 0.3},
    {"task": "sentiment", "weight": 0.3},
]

# Create mixed dataset
def create_mixed_dataset(tasks):
    datasets = []
    for task in tasks:
        ds = load_dataset(task["dataset"])
        ds = ds.map(format_functions[task["task"]])
        # Sample according to weight
        ds = ds.select(range(int(len(ds) * task["weight"])))
        datasets.append(ds)
    
    return concatenate_datasets(datasets)
```

### Few-Shot In-Context Learning

Include examples in the prompt:

```python
few_shot_template = """Given the following examples:

Example 1:
Input: {example1_input}
Output: {example1_output}

Example 2:
Input: {example2_input}
Output: {example2_output}

Now complete:
Input: {input}
Output:"""
```

### Chain-of-Thought (CoT) Prompting

Teach models to show reasoning:

```python
cot_template = """### Instruction:
Solve the following problem and explain your reasoning step by step.

### Problem:
{problem}

### Solution:
Let's think step by step:
1. {step1}
2. {step2}
3. {step3}

Therefore, the answer is: {answer}"""
```

---

## 📈 Evaluation Strategies

### Quantitative Metrics

```python
# ROUGE for summarization
from evaluate import load

rouge = load("rouge")
results = rouge.compute(predictions=preds, references=refs)

# BLEU for translation
bleu = load("bleu")
results = bleu.compute(predictions=preds, references=refs)

# Exact Match for Q&A
exact_match = load("exact_match")
results = exact_match.compute(predictions=preds, references=refs)
```

### Qualitative Analysis

```python
# Test on diverse prompts
test_prompts = [
    "Summarize the key points from this article...",
    "What is the sentiment of: 'Stock prices soared today'?",
    "Explain why the sky is blue.",
]

for prompt in test_prompts:
    output = model.generate(prompt, max_length=100)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}\n")
```

---

## 🐛 Troubleshooting

### Issue: Model Outputs Gibberish

**Solutions**:
```python
# 1. Check data formatting
print(dataset[0])  # Verify structure

# 2. Adjust temperature
generation_config = GenerationConfig(
    temperature=0.7,  # Lower = more focused
    top_p=0.9,
    repetition_penalty=1.2,
)

# 3. Train longer
num_train_epochs=5
```

### Issue: Model Doesn't Follow Instructions

**Solutions**:
```python
# 1. Use clearer instruction markers
### Instruction:
### Input:
### Response:

# 2. Increase dataset size (10K+ examples recommended)

# 3. Try a model pre-trained on instructions (FLAN-T5, Llama-2-Chat)
```

### Issue: Overfitting

**Solutions**:
```python
# 1. Add regularization
weight_decay=0.01
dropout=0.1

# 2. Early stopping
load_best_model_at_end=True
metric_for_best_model="eval_loss"

# 3. Data augmentation
# Paraphrase instructions, shuffle examples
```

---

## 📚 Additional Resources

### Papers
- [FLAN: Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- [InstructGPT: Training language models to follow instructions](https://arxiv.org/abs/2203.02155)
- [T5: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)

### Datasets
- [Hugging Face Datasets Hub](https://huggingface.co/datasets)
- [Awesome Instruction Datasets](https://github.com/yaodongC/awesome-instruction-dataset)

### Tutorials
- [Hugging Face Instruction Tuning Guide](https://huggingface.co/blog/instruction-tuning)
- [FLAN-T5 Tutorial](https://huggingface.co/docs/transformers/model_doc/flan-t5)

---

## 🎯 Next Steps

- Explore [RLHF (Reinforcement Learning from Human Feedback)](https://huggingface.co/blog/rlhf)
- Learn about [Reasoning Fine-Tuning](../04-Reasoning-Tuning/)
- Build a [multi-task instruction model](https://arxiv.org/abs/2204.07705)

---

## 💬 Need Help?

- Check the [SETUP.md](../SETUP.md) guide
- Review the [Troubleshooting section](#-troubleshooting)
- Open an issue on GitHub

