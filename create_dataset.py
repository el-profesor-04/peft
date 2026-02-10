import asyncio
import json
import os
import random
import re
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# 3rd party imports
from pdf2image import convert_from_path
from PIL import Image
from google import genai
from google.api_core import exceptions

load_dotenv()

# ================= CONFIGURATION ================= #

PDF_PATH = "data/principles-and-techniques-in-combinatronics-solutions-manual.pdf"

# Files
INTERMEDIATE_FILE = "data/intermediate_raw.json" # Safe save
FINAL_FILE = "data/combinatorics_dataset_merged.json" # Merged result
FAILED_LOG = "data/failed_pages.json" # Debug info

# Ranges (Inclusive)
RANGES = [
    (10, 91),   # Questions
    (92, 478)   # Solutions
]
QUESTIONS = RANGES[0]
SOLUTIONS = RANGES[1]


MAX_CONCURRENT_REQUESTS = 100
MAX_RETRIES = 3 
MODEL_NAME = "gemini-3-pro-preview" # "gemini-3-pro-preview" "gemini-3-flash-preview"

# Improved Prompt to reduce JSON errors
SYSTEM_PROMPT = """
Extract math problems or solutions from this book page image into JSON.

STRICT RULES:
1. Return ONLY valid JSON. No Markdown block (```json). No comments.
2. If a sentence from the PREVIOUS page continues at the very top of this page, put that text in key "prev_page".
3. Use the Question Number as the key (e.g. "43", "12(a)"). 
4. The Value must be the EXACT text/LaTeX.
5. If the page is a Table of Contents, Title Page, or Blank, return {"ignore": true}.
6. JSON strings cannot contain literal newlines. You must escape them as \n.
7. JSON requires you to escape internal quotes with a backslash: \".
8. The main question number is an integer like 25. or 12. and there may be sub-parts like (a) or (i) or (1) etc those must be merged with the main question and shouldnt be separate keys.
For e.g if there is (i) and (ii) for Question 12. then the key should just be "12" and should contain (i) and (ii) inside.
If the main question is not shown on the page, just assume its on the previous page and use the prev_page key.

FORMAT EXAMPLE:
{
  "exercise": 3,
  "prev_page": "(continuing text from previous page as there's no question number for a particular blurb...)",
  "14": "(Calculate the value of...)",
  "15": "(Prove that...)"
}
"""

# ================================================= #

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def clean_json_text(text: str) -> str:
    """Aggressive JSON cleanup."""
    text = text.strip()
    # Remove markdown code blocks
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()

async def process_page(sem: asyncio.Semaphore, page_num: int) -> Tuple[Dict, Dict]:
    """
    Returns (Success_Data, Failure_Data).
    One of them will be None.
    """
    async with sem:
        raw_response_text = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # 1. Render
                images = await asyncio.to_thread(
                    convert_from_path, PDF_PATH, dpi=150, 
                    first_page=page_num, last_page=page_num, fmt='jpeg'
                )
                
                if not images:
                    return None, {"page": page_num, "error": "Empty Render"}

                img = images[0]
                if img.height > 1024:
                    scale = 1024 / img.height
                    img = img.resize((int(img.width * scale), 1024))

                # 2. API Call
                print(f"🚀 P{page_num} (Att {attempt})...")
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=[SYSTEM_PROMPT, img],
                    config={"temperature": 0.1, "response_mime_type": "application/json"} 
                )
                
                raw_response_text = response.text

                # 3. Parse
                clean_text = clean_json_text(raw_response_text)
                data = json.loads(clean_text)
                
                if data.get("ignore") is True:
                    print(f"⚠️ P{page_num} Ignored (TOC/Blank).")
                    return None, None # Not an error, just skip

                data["_source_page"] = page_num
                print(f"✅ P{page_num} Success.")
                return data, None

            except Exception as e:
                print(f"⚠️ P{page_num} Error: {e}")
                if attempt == MAX_RETRIES:
                    return None, {
                        "page": page_num, 
                        "error": str(e), 
                        "raw_response": raw_response_text 
                    }
                await asyncio.sleep(random.uniform(1, 3))

        return None, None

def merge_dataset(raw_data: List[Dict]) -> List[Dict]:
    """
    Stitches 'prev_page' content to the last key of the previous page.
    """
    print("\n🧵 Merging pages...")
    
    # 1. Sort by page number
    sorted_pages = sorted(raw_data, key=lambda x: x["_source_page"])
    merged_output = []

    q_data = {}
    a_data = {}
    q_exercise = a_exercise = ""
    last = None

    for page in sorted_pages:
        if QUESTIONS[0]<=page["_source_page"]<=QUESTIONS[1]:
            if "exercise" in page:
                q_exercise = str(page["exercise"])
            if q_exercise=="":
                print(f"⚠️ P{page['_source_page']} has no exercise key or empty exercise. Skipping.")
                return {}
            for key in page:
                if key not in ["exercise", "_source_page", "prev_page"]:
                    q_data[q_exercise+"_"+key] = page[key]
                    last = q_exercise+"_"+key
                elif key == "prev_page" and last is not None:
                    q_data[last] += " " + page[key]
        elif SOLUTIONS[0]<=page["_source_page"]<=SOLUTIONS[1]:
            if "exercise" in page:
                a_exercise = str(page["exercise"])
            if a_exercise=="":
                print(f"⚠️ P{page['_source_page']} has no exercise key or empty exercise. Skipping.")
                return {}
            for key in page:
                if key not in ["exercise", "_source_page", "prev_page"]:
                    a_data[a_exercise+"_"+key] = page[key]
                    last = a_exercise+"_"+key
                elif key == "prev_page" and last is not None:
                    a_data[last] += " " + page[key]
    data = [{'id': k, 'question': q_data[k], 'answer': a_data.get(k, "")} for k in q_data.keys() if k in a_data.keys()]
    return data
                
        

    for i, current_page in enumerate(sorted_pages):
        page_content = current_page.copy()
        
        # Handle 'prev_page' content
        if "prev_page" in page_content:
            text_to_append = page_content.pop("prev_page") 
            
            if i > 0:
                prev_page_obj = sorted_pages[i-1]
                
                # Check continuity (Page 12 follows Page 11)
                if prev_page_obj["_source_page"] == current_page["_source_page"] - 1:
                    valid_keys = [k for k in prev_page_obj.keys() if k not in ["exercise", "_source_page", "prev_page"]]
                    
                    if valid_keys:
                        # Assuming insertion order roughly matches numeric order
                        last_key = list(valid_keys)[-1] 
                        print(f"   🔗 Linking P{current_page['_source_page']} start -> P{prev_page_obj['_source_page']} key '{last_key}'")
                        prev_page_obj[last_key] += " " + text_to_append
                    else:
                        print(f"   ⚠️ P{current_page['_source_page']} has cont. text, but P{prev_page_obj['_source_page']} has no keys.")
                else:
                    print(f"   ⚠️ Gap detected! P{prev_page_obj['_source_page']} -> P{current_page['_source_page']}. Cannot merge text.")
        
        merged_output.append(page_content)

    return merged_output

async def main():
    if not os.path.exists(PDF_PATH):
        print("❌ PDF not found")
        return

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # 1. Load Existing Data
    existing_data = []
    processed_pages = set()
    
    if os.path.exists(INTERMEDIATE_FILE):
        try:
            with open(INTERMEDIATE_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                processed_pages = {item["_source_page"] for item in existing_data}
            print(f"📂 Loaded {len(existing_data)} processed pages from {INTERMEDIATE_FILE}")
        except json.JSONDecodeError:
            print("⚠️ Intermediate file corrupted. Starting fresh.")
    
    # 2. Identify Missing Pages
    all_pages_in_range = []
    for start, end in RANGES:
        all_pages_in_range.extend(range(start, end + 1))
        
    pages_to_process = [p for p in all_pages_in_range if p not in processed_pages]
    
    if not pages_to_process:
        print("✨ All pages already processed! Skipping straight to merge.")
    else:
        print(f"🔎 Found {len(pages_to_process)} missing pages. Running {MODEL_NAME}...")
        
        # 3. Process only missing pages
        tasks = [process_page(sem, p) for p in pages_to_process]
        results = await asyncio.gather(*tasks)

        new_success_data = []
        failed_data = []

        for success, failure in results:
            if success:
                new_success_data.append(success)
            if failure:
                failed_data.append(failure)
        
        # Combine old + new
        total_data = existing_data + new_success_data
        
        # Save Intermediate
        with open(INTERMEDIATE_FILE, "w", encoding="utf-8") as f:
            json.dump(total_data, f, indent=2, ensure_ascii=False)
        
        # Save Failures
        if failed_data:
            print(f"\n❌ {len(failed_data)} pages failed. Check {FAILED_LOG}")
            with open(FAILED_LOG, "w", encoding="utf-8") as f:
                json.dump(failed_data, f, indent=2, ensure_ascii=False)
        else:
            print("\n✅ No failures in this run!")
            
        existing_data = total_data # Update for merging phase

    # --- Phase 2: Merge ---
    if existing_data:
        final_data = merge_dataset(existing_data)
        
        with open(FINAL_FILE, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        print(f"🎉 Final merged dataset ({len(final_data)} pages) saved to: {FINAL_FILE}")
    else:
        print("⚠️ No data to merge.")

if __name__ == "__main__":
    asyncio.run(main())