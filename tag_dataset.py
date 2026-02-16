import os
import json
import asyncio
from google import genai
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

CLASSIFY_PROMPT = """Look at this combinatorics problem and its answer. Classify the answer type as exactly one of:
- "numeric": the answer is a specific integer or decimal number
- "symbolic": the answer is an algebraic expression with variables (like n, m, k)
- "proof": the question asks to prove/show/demonstrate something, answer is an argument

Problem: {problem}
Answer: {answer}

Output only one word: numeric, symbolic, or proof"""

async def classify_sample(sample, semaphore):
    async with semaphore:
        # Extract the question from the HF message format
        question = sample['messages'][0]['content']
        # The 'answer' for classification is the assistant's response (including <think>)
        answer = sample['messages'][1]['content']
        
        try:
            response = await client.aio.models.generate_content(
                model="gemini-3-flash-preview",
                contents=CLASSIFY_PROMPT.format(problem=question, answer=answer)
            )
            tag = response.text.strip().lower()
            # Clean up just in case Gemini gets chatty
            if "numeric" in tag: tag = "numeric"
            elif "symbolic" in tag: tag = "symbolic"
            elif "proof" in tag: tag = "proof"
            else: tag = "numeric" # Default fallback
            
            sample['answer_type'] = tag
            return sample
        except Exception as e:
            sample['answer_type'] = "numeric"
            return sample

async def main():
    for split in ['train', 'val']:
        file_path = f"data/cot/{split}.jsonl"
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        semaphore = asyncio.Semaphore(20)
        tasks = [classify_sample(s, semaphore) for s in data]
        tagged_data = await tqdm.gather(*tasks)
        
        with open(f"data/cot/{split}_tagged.jsonl", 'w') as f:
            for item in tagged_data:
                f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    asyncio.run(main())