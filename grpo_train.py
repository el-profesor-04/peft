import re
import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from peft import PeftModel

# --- HELPER FUNCTIONS ---

def extract_all_numbers(text):
    """Finds all integers and decimals in a string, removing commas."""
    # Matches numbers like 1,000 or 45 or 235.5
    return re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text.replace(',', ''))

def normalize_symbolic(expr):
    """Basic normalization for symbolic expressions."""
    expr = expr.lower().strip()
    expr = re.sub(r'\s+', '', expr)
    expr = expr.replace('×', '*').replace('·', '*')
    expr = expr.replace('^', '**')
    return expr

# --- REWARD FUNCTIONS (The "User Logic" Stack) ---

def reward_format(completions, **kwargs):
    """Fundamental structure check."""
    return [1.0 if re.search(r"<think>.*?</think>", c, re.DOTALL) else 0.0 for c in completions]

def reward_numeric(output, ground_truth):
    """Handles single or multiple numerical answers."""
    predicted_nums = extract_all_numbers(output.split('</think>')[-1])
    gt_nums = extract_all_numbers(str(ground_truth))
    
    if not predicted_nums or not gt_nums:
        return 0.0
    
    # Check if all ground truth numbers appear in the prediction
    # This handles multi-part answers (i) 45 (ii) 235
    match_count = sum(1 for gt in gt_nums if gt in predicted_nums)
    if match_count == len(gt_nums):
        return 2.0
    elif match_count > 0:
        return 0.5 * (match_count / len(gt_nums)) # Partial credit
    return 0.0

def reward_symbolic(output, ground_truth):
    after_think = output.split('</think>')[-1].lower()
    gt_normalized = normalize_symbolic(ground_truth)
    
    variables = set(re.findall(r'[a-z]', gt_normalized))
    numbers = set(re.findall(r'\d+', gt_normalized))
    output_normalized = normalize_symbolic(after_think)
    
    vars_present = sum(1 for v in variables if v in output_normalized)
    nums_present = sum(1 for n in numbers if n in output_normalized)
    total = len(variables) + len(numbers)
    
    if total == 0: return 0.5
    return (vars_present + nums_present) / total * 1.5

def reward_proof(output, **kwargs):
    think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    if not think_match: return 0.0
    
    content = think_match.group(1).lower()
    indicators = ['therefore', 'thus', 'hence', 'since', 'assume', 'induction', 'qed']
    count = sum(1 for p in indicators if p in content)
    return min(count / 4, 1.0)

def reward_thinking_quality(output, answer_type, **kwargs):
    think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    if not think_match: return 0.0
    
    word_count = len(think_match.group(1).split())
    min_w = 30 if answer_type == "proof" else 20
    max_w = 600 if answer_type == "proof" else 400
    
    if word_count < min_w: return 0.0
    return 0.5 if word_count <= max_w else 0.2

# --- THE MASTER REWARD GATEKEEPER ---

def combined_reward_fn(completions, ground_truth, answer_type, **kwargs):
    """Dispatches rewards based on answer_type passed from dataset."""
    total_rewards = []
    
    # GRPOTrainer passes lists for completions and ground_truth
    for i in range(len(completions)):
        output = completions[i]
        gt = ground_truth[i]
        atype = answer_type[i]
        
        score = 0.0
        # 1. Format Check (Applied to all)
        score += reward_format([output])[0] * 0.5 
        
        # 2. Logic Check (Branched)
        if atype == "numeric":
            score += reward_numeric(output, gt)
        elif atype == "symbolic":
            score += reward_symbolic(output, gt)
        elif atype == "proof":
            score += reward_proof(output)
            
        # 3. Quality Check
        score += reward_thinking_quality(output, atype)
        
        total_rewards.append(score)
    return total_rewards


def format_grpo_prompt(example):
    return {
        "prompt": f"Solve this combinatorics problem step by step.\n\n{example['messages'][0]['content']}",
        "ground_truth": example['messages'][1]['content'].split('</think>')[-1].strip(),
        "answer_type": example['answer_type']
    }

if __name__ == "__main__":

    SFT_ADAPTER_PATH = "./adapters/sft-r64-final"
    RANK = 64
    BASE_MODEL_ID = "deepseek-ai/deepseek-math-7b-base"

    dataset = load_dataset("json", data_files="data/cot/train_tagged.jsonl")
    grpo_dataset = dataset['train'].filter(lambda x: x['answer_type'] != 'proof').map(format_grpo_prompt)

    training_args = GRPOConfig(
        output_dir="./grpo-combinatorics-final",
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=8, # Group size
        max_completion_length=1024,
        bf16=True,
        report_to="wandb",
        logging_steps=1,
        # A6000 optimization:
        use_vllm = True, # Highly recommended for GRPO generation speed
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)

    trainer = GRPOTrainer(
        model=model,  # Point directly to your SFT adapter folder
        reward_funcs=[combined_reward_fn],
        args=training_args,
        train_dataset=grpo_dataset,
        # IMPORTANT: We keep the same LoRA config so we are "extending" the SFT training
    )

    trainer.train()