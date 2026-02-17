from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "deepseek-ai/deepseek-math-7b-base"
ADAPTER_PATH = "./grpo-combinatorics-final/checkpoint-591" # Use your final checkpoint
SAVE_PATH = "./final_merged_grpo_model"

print("Merging LoRA weights... this takes ~2 mins on A6000")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(SAVE_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}. You can now point eval.py to this folder.")