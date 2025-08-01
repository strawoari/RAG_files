import os
import asyncio
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import logging
from openai import AsyncAzureOpenAI
import aiofiles
import json
import re
import httpx
from evaluation import spacy_score

logging.basicConfig(level=logging.INFO)

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("LIGHT_LLM_ENDPOINT")

AZURE_EMBEDDING_ENDPOINT = os.getenv("LIGHT_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("OPENAI_API_VERSION")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs/lightrag1")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

async def rate_and_write_output(qa, contexts, index: int) -> None:
    query = qa['question']
    rewritten_query = qa['rewritten_query']
    file_path = OUTPUT_DIR / f"baseline_{type}.json"
    log = {
        "query": query,
        "rewritten_query": rewritten_query,
        "response": response,
        "contexts": contexts,
        "index": index
    }
    try:
        async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(log, ensure_ascii=False, indent=4) + "\n")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
    return

async def process_query(expanded_query: str, query: str, sem: asyncio.Semaphore) -> tuple[str, str]:
    # print(f"DEBUG: Processing query: {expanded_query}")
    url = "http://localhost:1024/query"
    payload = {
        "query": expanded_query, 
        "only_need_context": True,
        "only_need_prompt": False,
        "mode": "hybrid", 
        "top_k": 15,
        "max_token_for_global_context": 200,
        "max_token_for_local_context": 200,
        "response_type": "Single Paragraph"
    }
    headers = {"Content-Type": "application/json"}
    
    max_retries = 3
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            async with sem:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                res = response.json()["response"]
                print(f"result saved ({len(res)} chars)")
                if res.startswith("Error"):
                    return "Error in lightrag context retrieval API: " + res, []
                contexts = await get_contexts(res)
                return contexts
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, query '{query}': {e}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed, query '{query}': {e}")
                return f"Error in run_query: {str(e)}", []

def cosine_similarity(a, b):
    try:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}")
        return 0.0
embed_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_EMBEDDING_API_VERSION,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
)
async def get_contexts(context_response: str):
    # print(f"DEBUG: query: {query}\ncontext_response: {context_response}")
    try:
        entities_json_str = re.search(r"-----Entities\(KG\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        relationships_json_str = re.search(r"-----Relationships\(KG\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        docs_json_str = re.search(r"-----Document Chunks\(DC\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        entities_list = json.loads(entities_json_str.group(1)) if entities_json_str else []
        relationships_list = json.loads(relationships_json_str.group(1)) if relationships_json_str else []
        docs_list = json.loads(docs_json_str.group(1)) if docs_json_str else []
        return {"entities": entities_list, "relationships": relationships_list, "docs": docs_list}
    except Exception as e:
        logging.error(f"Error in get_ranked_contexts: {e}")
        return []

generator = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)
async def rewrite_query(query: str, sem: asyncio.Semaphore):
    prompt = query
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', query)
    english_chars = re.findall(r'[A-Za-z]', query)
    if len(chinese_chars) > len(english_chars):
        prompt = f"""You are a query translation assistant for a retrieval-based question answering system. Your task is to translate user queries written in Chinese into precise and natural English, in a way that aligns closely with entity names, relations, and terminology typically used in source documents (e.g., manuals, FAQs, policy pages).
Follow these guidelines:
1.Preserve the user's original intent and meaning.
2.Use formal or commonly used expressions that are likely to appear in the documents (e.g., translate “交电费” as “pay electricity bill” or “how to pay my electricity bill”).
3.Do not add new information or assumptions.
4.Clarify ambiguous references in the original question only if they can be clearly inferred from context (e.g., "这个" → "the electronic bill" if that’s clearly the subject).
5.If the original query is already clear, just improve fluency and precision without excessive rewriting.
6.Your priority is accurate translation, not expansion.

Original query (in Chinese):
{query}

Translated query (in English):
"""
    elif len(english_chars) > len(chinese_chars):
        prompt = f"""You are a query rewriting assistant for a retrieval-based question answering system. Your task is to rewrite user queries in a way that aligns closely with entity names, relations, and terminology typically used in source documents (e.g., manuals, FAQs, policy pages).
Follow these guidelines:
1.Preserve the user's original intent and meaning.
2.Use formal or commonly used expressions that are likely to appear in the documents (e.g., translate "pay electricity bill" as "how to pay my electricity bill").
3.Do not add new information or assumptions.
4.Clarify ambiguous references in the original question only if they can be clearly inferred from context (e.g., "the electronic bill" → "这个" if that’s clearly the subject).
5.If the original query is already clear, just improve fluency and precision without excessive rewriting.

Original query:
{query}

Rewritten query:
"""
    try:
        async with sem:
            completion = await generator.chat.completions.create(
                model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}])
            return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error expanding query: {e}")
        return query
        
async def main() -> None:
    sem = asyncio.Semaphore(5)
    
    async def handle_one_qa(qa, sem, index):
        try:
            rewritten_query = await rewrite_query(qa['question'], sem)
            contexts = await process_query(rewritten_query, sem)
            qa['rewritten_query'] = rewritten_query
            await rate_and_write_output(qa, contexts, index)
        except Exception as e:
            print(f"Error processing QA pair: {e}")
            
    coros = []
    async with aiofiles.open("/Users/amychan/rag_files/lightrag/outputs/v0/annotated.json", "r") as f:
        raw = await f.read()
    json_data = json.loads(raw)
    for _, data in enumerate(json_data):
        q = data['Query']
        a = data['Answer']
        index = data['index']
        coros.append(handle_one_qa({"question": q, "answer": a}, sem, index = index))
    await asyncio.gather(*coros)

if __name__ == "__main__":
    asyncio.run(main())