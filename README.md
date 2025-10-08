# LLM Finetuning Cookbook

🔥 A hands-on, code-first collection of recipes for fine-tuning Large Language Models. 🔥

This repository is not a polished library but a collection of practical, well-commented notebooks and scripts. The goal is to provide deep insights into the code and mechanics behind various LLM fine-tuning techniques, inspired by the fantastic work of practitioners like [Youssef Hosni](https://github.com/youssefHosni).

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
