# 🧠 Reasoning Fine-Tuning

Enhance language models' ability to perform logical, mathematical, and multi-step reasoning through specialized fine-tuning techniques.

---

## 📖 Overview

**What is Reasoning Fine-Tuning?**

Reasoning fine-tuning improves models' capabilities in:
- 🔢 **Mathematical reasoning**: Arithmetic, algebra, calculus
- 🧩 **Logical reasoning**: Deduction, induction, abduction
- 🔗 **Multi-step reasoning**: Breaking complex problems into steps
- 🎯 **Common sense reasoning**: Everyday knowledge and inference

**Why It's Challenging:**

Traditional fine-tuning often produces models that:
- ❌ Memorize answers instead of learning reasoning
- ❌ Struggle with problems requiring multiple steps
- ❌ Can't generalize to out-of-distribution problems
- ❌ Make arithmetic errors on simple calculations

**Advanced Techniques:**

- **Chain-of-Thought (CoT)**: Explicit step-by-step reasoning
- **Self-Consistency**: Generate multiple reasoning paths
- **Reinforcement Learning**: Reward correct reasoning processes
- **GRPO (Generalized Reward Policy Optimization)**: Advanced RL for reasoning

---

## 📓 Notebooks in This Section

### Math-Reasoning-Qwen-GRPO.ipynb

**Description**: Enhance the Qwen model's mathematical reasoning using GRPO (Generalized Reward Policy Optimization).

**Key Concepts:**
- Mathematical reasoning datasets (GSM8K, MATH)
- Chain-of-thought prompting
- Reinforcement learning for reasoning
- Unsloth for optimized training
- GRPO algorithm

**Use Case**: Math tutoring, problem-solving assistants, STEM education

**Dependencies:**
```bash
make install-reasoning

# Note: Unsloth may require special installation
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Requirements:**
- **GPU**: 24GB+ VRAM (A100/A6000 recommended)
- **Time**: 6-12 hours
- **Dataset**: GSM8K (8.5K training examples)

---

## 🔑 Key Techniques

### 1. Chain-of-Thought (CoT) Prompting

Teach models to show their reasoning:

```python
# Standard prompting (often fails)
prompt = "What is 15% of 80?"
# Output: "12" (correct but no reasoning)

# Chain-of-Thought prompting
prompt = """Question: What is 15% of 80?
Answer: Let's think step by step.
1. Convert 15% to decimal: 15/100 = 0.15
2. Multiply: 0.15 × 80 = 12
Therefore, 15% of 80 is 12."""
```

### 2. Dataset Formatting for Reasoning

```python
# GSM8K format
example = {
    "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
}

# Format for training
def format_example(example):
    return f"""### Question:
{example['question']}

### Answer:
Let's solve this step by step:
{example['answer']}"""
```

### 3. GRPO (Generalized Reward Policy Optimization)

Advanced RL technique for reasoning:

```python
from trl import GRPOTrainer, GRPOConfig

# Define reward function
def reward_fn(questions, answers, correct_answers):
    """
    Reward based on:
    1. Final answer correctness
    2. Reasoning quality
    3. Step clarity
    """
    rewards = []
    for ans, correct in zip(answers, correct_answers):
        # Extract final answer
        final = extract_answer(ans)
        # Check correctness
        if final == correct:
            reward = 1.0
            # Bonus for showing work
            if "step" in ans.lower():
                reward += 0.2
        else:
            reward = -0.5
        rewards.append(reward)
    return rewards

# Configure GRPO
grpo_config = GRPOConfig(
    learning_rate=1e-5,
    batch_size=32,
    num_train_epochs=3,
    reward_fn=reward_fn,
)

# Train
trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    train_dataset=dataset,
)
trainer.train()
```

---

## 📊 Popular Reasoning Datasets

| Dataset | Size | Difficulty | Focus |
|---------|------|------------|-------|
| [GSM8K](https://github.com/openai/grade-school-math) | 8.5K | Grade School | Arithmetic word problems |
| [MATH](https://github.com/hendrycks/math) | 12.5K | High School+ | Advanced math (algebra, calculus) |
| [AQuA-RAT](https://huggingface.co/datasets/aqua_rat) | 100K | GRE level | Algebraic reasoning |
| [StrategyQA](https://huggingface.co/datasets/wics/strategy-qa) | 2.7K | Multi-hop | Implicit reasoning steps |
| [CommonsenseQA](https://huggingface.co/datasets/commonsense_qa) | 12K | Common sense | Everyday reasoning |
| [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa) | 8.6K | Logic | Logical reasoning |

---

## 💡 Best Practices

### Prompt Engineering for Reasoning

```python
# Zero-shot CoT (simple but effective)
prompt = "Q: {question}\nA: Let's think step by step."

# Few-shot CoT (better for complex problems)
prompt = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
A: The cafeteria started with 23 apples. They used 20, so they have 23 - 20 = 3. They bought 6 more, so 3 + 6 = 9. The answer is 9.

Q: {question}
A: Let's think step by step."""

# Self-consistency (generate multiple answers, take majority)
answers = []
for _ in range(5):
    answer = model.generate(prompt, temperature=0.7)
    answers.append(extract_answer(answer))
final_answer = most_common(answers)
```

### Training Strategies

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Lower learning rate for reasoning
    learning_rate=5e-6,
    
    # More epochs for reasoning tasks
    num_train_epochs=5,
    
    # Smaller batch size (complex examples)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    
    # Warmup helps stabilize training
    warmup_ratio=0.1,
    
    # Cosine scheduling works well
    lr_scheduler_type="cosine",
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    
    # Load best model
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
```

### Answer Extraction

```python
import re

def extract_answer(text):
    """Extract final answer from reasoning text."""
    
    # Method 1: Look for "#### answer" (GSM8K format)
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return int(match.group(1))
    
    # Method 2: Look for "The answer is X"
    match = re.search(r'[Tt]he answer is (\d+)', text)
    if match:
        return int(match.group(1))
    
    # Method 3: Last number in text
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[-1])
    
    return None
```

---

## 🚀 Advanced Techniques

### Self-Consistency Decoding

Generate multiple reasoning paths and take the majority vote:

```python
def self_consistency_generate(model, prompt, n=5):
    """Generate multiple answers and take majority."""
    answers = []
    
    for _ in range(n):
        output = model.generate(
            prompt,
            temperature=0.7,  # Higher temp for diversity
            max_length=512,
        )
        answer = extract_answer(output)
        answers.append(answer)
    
    # Take majority vote
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

### Least-to-Most Prompting

Break complex problems into simpler sub-problems:

```python
def least_to_most_prompt(question):
    """Decompose problem into steps."""
    
    # Step 1: Decomposition
    decomp_prompt = f"""Break down the following problem into simpler sub-problems:
Question: {question}
Sub-problems:"""
    
    subproblems = model.generate(decomp_prompt)
    
    # Step 2: Solve each sub-problem
    solutions = []
    for subproblem in subproblems:
        sol_prompt = f"Solve: {subproblem}"
        solution = model.generate(sol_prompt)
        solutions.append(solution)
    
    # Step 3: Combine solutions
    final_prompt = f"""Given these solutions to sub-problems:
{solutions}

Solve the original problem:
{question}"""
    
    return model.generate(final_prompt)
```

### Verifier-Based Training

Train a separate model to verify answers:

```python
# Train verifier
verifier = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,  # Correct/Incorrect
)

# Use verifier during generation
def verified_generate(generator, verifier, prompt):
    """Generate and verify multiple candidates."""
    candidates = []
    
    for _ in range(10):
        output = generator.generate(prompt)
        score = verifier(prompt + output)  # Correctness score
        candidates.append((output, score))
    
    # Return highest-scored candidate
    return max(candidates, key=lambda x: x[1])[0]
```

---

## 📈 Evaluation Metrics

### Accuracy

```python
def compute_accuracy(predictions, references):
    """Exact match accuracy."""
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)
```

### Reasoning Quality

```python
def evaluate_reasoning_quality(answer):
    """Score reasoning quality (0-1)."""
    score = 0.0
    
    # Has step-by-step reasoning?
    if any(word in answer.lower() for word in ["step", "first", "then", "therefore"]):
        score += 0.3
    
    # Shows calculations?
    if re.search(r'\d+\s*[+\-*/]\s*\d+', answer):
        score += 0.3
    
    # Explains logic?
    if len(answer.split()) > 20:  # Detailed explanation
        score += 0.2
    
    # Has final answer?
    if extract_answer(answer) is not None:
        score += 0.2
    
    return score
```

### Human Evaluation

```python
# Template for human eval
eval_template = {
    "question": "...",
    "model_answer": "...",
    "correct_answer": "...",
    "ratings": {
        "correctness": 0,      # 0-5
        "clarity": 0,          # 0-5
        "completeness": 0,     # 0-5
        "efficiency": 0,       # 0-5 (conciseness)
    }
}
```

---

## 🐛 Troubleshooting

### Issue: Model Makes Arithmetic Errors

**Solutions**:
```python
# 1. Use calculator tool
def augment_with_calculator(text):
    # Find arithmetic expressions
    expressions = re.findall(r'(\d+\s*[+\-*/]\s*\d+)', text)
    for expr in expressions:
        result = eval(expr)  # Be careful with eval!
        text = text.replace(expr, f"{expr} = {result}")
    return text

# 2. Train with more arithmetic examples

# 3. Use symbolic math library
from sympy import sympify
result = sympify("15 * 3 + 7").evalf()
```

### Issue: Model Gives Short, Incorrect Answers

**Solutions**:
```python
# 1. Enforce step-by-step in prompt
prompt = "Q: {question}\nA: Let's solve this step by step:\n1."

# 2. Add length penalty
generation_config = GenerationConfig(
    min_length=50,  # Force longer outputs
    length_penalty=1.5,
)

# 3. Use few-shot examples showing detailed reasoning
```

### Issue: Poor Generalization

**Solutions**:
```python
# 1. Diversify training data
# Include problems from multiple domains

# 2. Data augmentation
# Rephrase questions, change numbers

# 3. Curriculum learning
# Start with easy problems, gradually increase difficulty
trainer = Trainer(
    callbacks=[CurriculumCallback(difficulty_schedule)]
)
```

---

## 📚 Additional Resources

### Papers
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
- [Let's Verify Step by Step (Process Supervision)](https://arxiv.org/abs/2305.20050)
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

### Tools
- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM training
- [vLLM](https://github.com/vllm-project/vllm) - Efficient inference
- [Guidance](https://github.com/guidance-ai/guidance) - Structured generation

### Benchmarks
- [BIG-Bench](https://github.com/google/BIG-bench)
- [MMLU](https://github.com/hendrycks/test)
- [HumanEval](https://github.com/openai/human-eval) (code reasoning)

---

## 🎯 Next Steps

- Explore **Tool Use**: Integrate calculators, code executors
- Learn **Multi-Modal Reasoning**: Combine text, images, tables
- Build **Interactive Tutors**: Real-time reasoning assistance
- Research **Neurosymbolic AI**: Combine neural nets with symbolic reasoning

---

## 💬 Need Help?

- Check the [SETUP.md](../SETUP.md) guide
- Review the [Troubleshooting section](#-troubleshooting)
- Open an issue on GitHub
- Join discussions in the community

