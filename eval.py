import os
import json
import re
import time
import torch
import asyncio
from tqdm import tqdm
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
GEMINI_MODEL = "gemini-3-flash-preview" 
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

BASE_MODEL_ID = "deepseek-ai/deepseek-math-7b-base"
TEST_FILE = "data/cot/test_tagged.jsonl"

MODELS_TO_TEST = {
    "Base-ZeroShot": None,  # <--- Added this
    #"SFT-Rank8": "./adapters/sft-r8-final",
    #"SFT-Rank64": "./adapters/sft-r64-final",
    #"GRPO-Final": "./grpo-combinatorics-final/checkpoint-591"
}

# --- ROBUST EXTRACTION HELPERS ---

def extract_answer_aggressive(text):
    if "</think>" in text:
        candidate = text.split("</think>")[-1].strip()
        if len(candidate) > 0:
            return candidate
    
    patterns = [
        r"boxed\{(.+?)\}",   
        r"answer is (.+?)(?:\.|$)", 
        r"#### (.+?)(?:\.|$)"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1)
            
    return text[-100:]

def extract_numbers(text):
    return re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text.replace(',', ''))

def normalize_symbolic(expr):
    expr = str(expr).lower().strip()
    expr = re.sub(r'\s+', '', expr)
    expr = expr.replace('×', '*').replace('·', '*').replace('^', '**')
    return expr

def symbolic_overlap_score(output, gt_norm):
    variables = set(re.findall(r'[a-z]', gt_norm))
    numbers = set(re.findall(r'\d+', gt_norm))
    output_norm = normalize_symbolic(output)
    
    vars_present = sum(1 for v in variables if v in output_norm)
    nums_present = sum(1 for n in numbers if n in output_norm)
    total = len(variables) + len(numbers)
    return (vars_present + nums_present) / total if total > 0 else 0.5

# --- GEMINI JUDGE ---

async def call_gemini_judge(problem, model_output, answer_type, ground_truth):
    if answer_type == "proof":
        criteria = "Is the proof logic valid? Are steps complete?"
    elif answer_type == "symbolic":
        criteria = f"Is the derived expression equivalent to: {ground_truth}?"
    else:
        criteria = f"Is the final numeric answer equal to: {ground_truth}?"

    prompt = f"""You are a strict math grader.
Problem: {problem}
Student Answer: {model_output}
Task: {criteria}
Rate the response 1-10 based on reasoning quality and correctness.
1 = Completely wrong.
10 = Perfect reasoning and correct answer.
Output ONLY the integer score."""

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt
        )
        score = int(re.search(r'\d+', response.text).group())
        return min(max(score, 1), 10) 
    except:
        return None

# --- EVALUATOR LOGIC ---

async def load_model_and_generate(model_name, adapter_path, test_data):
    print(f"\n🔄 Loading {model_name}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    model.eval()
    
    results = []
    gemini_tasks = []
    
    print(f"📝 Generating answers for {len(test_data)} items...")
    
    for ex in tqdm(test_data):
        user_query = ex['messages'][0]['content']
        prompt = f"User: {user_query}\nAssistant: <think>"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Assistant: <think>" in full_output:
            generated_text = full_output.split("Assistant: <think>")[-1]
            final_output_for_judge = "<think>" + generated_text
        else:
            final_output_for_judge = full_output

        final_ans_text = extract_answer_aggressive(final_output_for_judge)
        
        gt_text = ex['messages'][1]['content'].split('</think>')[-1].strip()
        atype = ex['answer_type']
        
        is_numeric_correct = False
        symbolic_score = 0.0
        
        if atype == "numeric":
            p_nums = extract_numbers(final_ans_text)
            g_nums = extract_numbers(gt_text)
            if g_nums and p_nums:
                is_numeric_correct = all(n in p_nums for n in g_nums)
        
        elif atype == "symbolic":
            symbolic_score = symbolic_overlap_score(final_ans_text, normalize_symbolic(gt_text))
        
        gemini_tasks.append(
            call_gemini_judge(user_query, final_output_for_judge, atype, gt_text)
        )
        
        results.append({
            "type": atype,
            "numeric_correct": is_numeric_correct,
            "symbolic_score": symbolic_score,
            "gemini_score": None
        })

    gemini_scores = await asyncio.gather(*gemini_tasks)

    for r, score in zip(results, gemini_scores):
        r["gemini_score"] = score

    del model
    del base_model
    torch.cuda.empty_cache()
    
    return results

# --- MAIN ---

async def main():
    if not os.path.exists(TEST_FILE):
        print(f"❌ Error: Test file not found at {TEST_FILE}")
        return

    with open(TEST_FILE, 'r') as f:
        test_data = [json.loads(line) for line in f]

    final_table_data = []

    for name, path in MODELS_TO_TEST.items():
       
            
        res = await load_model_and_generate(name, path, test_data)
        
        numerics = [r for r in res if r['type'] == 'numeric']
        num_acc = sum(1 for r in numerics if r['numeric_correct']) / len(numerics) if numerics else 0.0
        
        symbolics = [r for r in res if r['type'] == 'symbolic']
        sym_score = sum(r['symbolic_score'] for r in symbolics) / len(symbolics) if symbolics else 0.0
        
        all_gemini = [r['gemini_score'] for r in res if r['gemini_score']]
        overall_avg = sum(all_gemini) / len(all_gemini) if all_gemini else 0.0
        
        proofs = [r for r in res if r['type'] == 'proof']
        proof_scores = [r['gemini_score'] for r in proofs if r['gemini_score']]
        proof_avg = sum(proof_scores) / len(proof_scores) if proof_scores else 0.0
        
        final_table_data.append({
            "Model": name,
            "Numeric Acc": f"{num_acc:.1%}",
            "Symbolic Score": f"{sym_score:.2f}",
            "Proof (Gemini)": f"{proof_avg:.1f}/10",
            "Overall Gemini": f"{overall_avg:.1f}/10"
        })

    print("\n\n")
    headers = ["Model", "Numeric Acc", "Symbolic Score", "Proof (Gemini)", "Overall Gemini"]
    widths = [20, 15, 15, 15, 15]
    
    header_row = "".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header_row)
    print("-" * len(header_row))
    
    for row in final_table_data:
        print("".join(f"{row[h]:<{w}}" for h, w in zip(headers, widths)))
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
