# 🧮 DeepSeek-Math Combinatorics: SFT vs. GRPO

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![TRL](https://img.shields.io/badge/TRL-GRPO-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

This project investigates the **Reasoning Gap** in Large Language Models by fine-tuning **DeepSeek-Math-7B** on a specialized combinatorics dataset. We compare the efficacy of **Supervised Fine-Tuning (SFT)** against **Reinforcement Learning (GRPO)** to determine if RL is necessary to teach a model to "think" mathematically rather than just mimic the style of a solution.

## 📊 Key Findings

Our experiments reveal a clear distinction between learning *syntax* (style) and *semantics* (logic).

| Model | Numeric Acc | Symbolic Score | Proof (Gemini Judge) | Overall Gemini |
| :--- | :--- | :--- | :--- | :--- |
| **Base-ZeroShot** | **0.0%** | 0.49 | 1.0/10 | 2.1/10 |
| **SFT (Rank 8)** | **0.0%** | 0.84 | **1.6/10** | 1.9/10 |
| **SFT (Rank 64)** | **0.0%** | **0.88** | 1.4/10 | 1.8/10 |
| **GRPO (RL)** | **9.1%** | **0.88** | 1.5/10 | 1.8/10 |

### 🧠 The Narrative
1.  **The Novice (Base Model):** Fails completely at the task. It lacks both the vocabulary of combinatorics ($0.49$ symbolic score) and the ability to solve problems ($0.0\%$ numeric).
2.  **The Mimic (SFT):** Supervised Fine-Tuning successfully teaches the model to *sound* like a mathematician. The symbolic score jumps to **0.88**, meaning it uses the correct notation and structure. However, it still has **zero** problem-solving ability ($0.0\%$ numeric).
3.  **The Thinker (GRPO):** Reinforcement Learning was the **only** method to unlock actual calculation capabilities. By optimizing against a correctness reward, it achieved a **9.1%** numeric accuracy—an infinite improvement over the SFT baseline.

> **Conclusion:** SFT is necessary to learn the *domain language*, but insufficient for *reasoning*. Only Reinforcement Learning (GRPO) forced the model to verify its own logic.

---

## 🛠️ Methodology

### 1. Dataset
We utilized a custom subset of high-difficulty combinatorics problems formatted for **Chain-of-Thought (CoT)** reasoning.
* **Format:** `User: <Problem>\nAssistant: <think>...reasoning...</think> <Answer>`
* **Size:** ~500 high-quality samples.

### 2. Supervised Fine-Tuning (SFT)
We trained two LoRA (Low-Rank Adaptation) adapters to establish a baseline.
* **Base Model:** `deepseek-ai/deepseek-math-7b-base`
* **Ranks:** `r=8` vs `r=64`
* **Target Modules:** `q_proj`, `v_proj`, `up_proj`, `down_proj`
* **Objective:** Minimize Cross-Entropy Loss on the ground truth reasoning traces.

### 3. Group Relative Policy Optimization (GRPO)
We initialized the RL phase using the **SFT (Rank 64)** adapter.
* **Algorithm:** GRPO (DeepSeek-R1 paper).
* **Group Size:** 8 generations per prompt.
* **Reward Functions:**
    1.  **Format Reward:** Strict adherence to `<think>` and `</think>` tags.
    2.  **Numeric Reward:** Exact match with ground truth integers.
    3.  **Thinking Quality:** Penalties for empty or overly short reasoning chains.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/combinatorics-grpo.git](https://github.com/yourusername/combinatorics-grpo.git)
cd combinatorics-grpo

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention (Optional but recommended for A6000/A100)
pip install flash-attn --no-build-isolation