import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb
import os

# --- CONFIGURATION ---
MODEL_ID = "deepseek-ai/deepseek-math-7b-base"
TRAIN_FILE = "data/cot/train.jsonl"
VAL_FILE = "data/cot/val.jsonl"
RANK = 8  # 64 v 8
OUTPUT_DIR = f"./checkpoints/sft-r{RANK}"

# 1. Initialize WandB
wandb.init(project="combinatorics-sft", name=f"deepseek-math-r{RANK}")

# 2. Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Standard for SFT training

# 3. Model Setup (Optimized for A6000)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # Uses the Flash Attention 2 we installed
    attn_implementation="flash_attention_2" 
)
model.config.use_cache = False  # Disable for training

# 4. LoRA Setup (Targets MLPs for math reasoning)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=RANK,
    lora_alpha=RANK * 2,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
)

def tokenize_and_mask(example):
    # We construct the full conversation manually
    prompt_text = f"User: {example['messages'][0]['content']}\nAssistant: "
    completion_text = f"{example['messages'][1]['content']}{tokenizer.eos_token}"
    
    # Tokenize separately to find lengths
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    
    # Combine them
    input_ids = prompt_ids + completion_ids
    # Mask labels: -100 for prompt, actual IDs for completion
    labels = ([-100] * len(prompt_ids)) + completion_ids
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }

    
# We use a custom formatting function to match the chat structure
def formatting_prompts_func(example):
    # Instead of one long string, we return a dict of lists
    prompts = []
    completions = []
    for msg_list in example['messages']:
        # msg_list[0] is User, msg_list[1] is Assistant
        prompts.append(f"User: {msg_list[0]['content']}\nAssistant: ")
        completions.append(f"{msg_list[1]['content']}{tokenizer.eos_token}")
    
    return {"prompt": prompts, "completion": completions}

# IMPORTANT: Matching the template to the formatting function above
response_template = "\nAssistant: "

# 6. Load Datasets
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "eval": VAL_FILE})

# 7. Training Arguments
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=50,
    dataset_text_field="input_ids",
    report_to="wandb",
    gradient_checkpointing=True, # Saves VRAM at slight speed cost
)

train_dataset = dataset["train"].map(tokenize_and_mask, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["eval"].map(tokenize_and_mask, remove_columns=dataset["eval"].column_names)

# 8. Trainer Execution
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
)

# Sanity Check: Verify the data collator doesn't mask everything
# --- THE REAL SANITY CHECK ---
print("\n--- FINAL VERIFICATION ---")
sample = train_dataset[0]
print(f"Total tokens: {len(sample['input_ids'])}")
print(f"Masked tokens (-100 count): {sample['labels'].count(-100)}")
first_unmasked_idx = next(i for i, x in enumerate(sample['labels']) if x != -100)
print(f"First unmasked token ID: {sample['labels'][first_unmasked_idx]}")
print(f"First unmasked word: '{tokenizer.decode([sample['labels'][first_unmasked_idx]])}'")


trainer.train()

# 9. Save final adapter
trainer.save_model(f"./adapters/sft-r{RANK}-final")