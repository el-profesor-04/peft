import os
import json
import asyncio
import re
import random
from google import genai
from google.api_core import exceptions
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
INPUT_FILE = "data/combinatorics_dataset_merged.json"
OUTPUT_DIR = "data/cot"
CONCURRENCY_LIMIT = 50  # Number of parallel tasks
MODEL_NAME = "gemini-3-flash-preview"

# Initialize the new Client
client = genai.Client(api_key=GOOGLE_API_KEY)

REFORMAT_PROMPT = """You are a mathematics tutor. Given a combinatorics problem and its solution, rewrite the solution as a detailed chain-of-thought reasoning inside <tool_call> tags, followed by a clean final answer. 

Rules:
- Inside <think>: show all intermediate steps, identify what combinatorial technique applies (permutations, combinations, etc.), and compute step by step.
- If the problem has multiple parts (e.g., (i) and (ii)), address all parts clearly within the same <think> block.
- After </think>: state the final answer clearly for all parts. Use the format: "The answer is [value]." or "The answers are (i) [val1], (ii) [val2]."
- Do not add any text between <think> and the problem restatement.
- If the original solution has an error, correct it.

Problem: {problem}
Original Solution: {solution}

Output the reformatted solution now:"""

# --- UTILS ---
def validate_output(text):
    """Checks for exactly one think block and a parseable answer."""
    has_start = "<think>" in text
    has_end = "</think>" in text
    answer_part = text.split("</think>")[-1] if has_end else ""
    has_answer = bool(re.search(r'\d+', answer_part))
    return has_start and has_end and has_answer

async def process_sample(sample, semaphore):
    """Calls Gemini asynchronously to reformat a single sample."""
    async with semaphore:
        prompt = REFORMAT_PROMPT.format(
            problem=sample['question'], 
            solution=sample['answer']
        )
        
        try:
            # Using the .aio property for asynchronous calls
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            content = response.text.strip()
            
            if validate_output(content):
                return {
                    "id": sample['id'],
                    "question": sample['question'],
                    "reformatted_solution": content
                }
            return None
        except Exception as e:
            # Handle rate limits or temporary errors
            if "429" in str(e):
                await asyncio.sleep(2) # Backoff
                return await process_sample(sample, semaphore)
            print(f"Error on {sample['id']}: {e}")
            return None

def format_to_hf(example):
    """Converts to HuggingFace Chat Format."""
    return {
        "messages": [
            {
                "role": "user", 
                "content": f"Solve this combinatorics problem step by step.\n\n{example['question']}"
            },
            {
                "role": "assistant",
                "content": example['reformatted_solution']
            }
        ]
    }

# --- MAIN EXECUTION ---
async def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples. Starting async reformatting...")
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [process_sample(sample, semaphore) for sample in data]
    
    reformatted_results = []
    # tqdm.gather handles the progress bar for async tasks
    results = await tqdm.gather(*tasks)
    
    reformatted_data = [r for r in results if r is not None]
    print(f"Successfully reformatted {len(reformatted_data)}/{len(data)} samples.")

    # Convert to HF format
    hf_dataset = [format_to_hf(d) for d in reformatted_data]

    # 80/10/10 Split (using your 489 count)
    random.seed(42)
    random.shuffle(hf_dataset)
    
    n = len(hf_dataset)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    splits = {
        'train': hf_dataset[:train_end],
        'val': hf_dataset[train_end:val_end],
        'test': hf_dataset[val_end:]
    }

    for name, content in splits.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        with open(path, 'w') as f:
            for item in content:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(content)} samples to {path}")

if __name__ == "__main__":
    asyncio.run(main())