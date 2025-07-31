
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

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

type = "hybrid"

embed_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_EMBEDDING_API_VERSION,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
)

async def rate_and_write_output(query: str, response: str, answer: str, source: str) -> None:
    split_result = answer.split('参考资料')
    main_content = response
    refs = ""
    if len(split_result) >= 2:
        main_content, refs = split_result[0], split_result[1]
    score = spacy_score(main_content, answer)
    file_path = OUTPUT_DIR / f"baseline_{type}.json"
    log = {
        "query": query,
        "answer": answer,
        "response": main_content,
        "score": score
    }
    try:
        async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(log, ensure_ascii=False, indent=4) + "\n")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
    return

async def run_query(mode: str, qa_pair: dict, sem: asyncio.Semaphore, user_prompt: str | None = None) -> str:
    """Executes *rag.query* in a thread and writes the result to file."""
    url = "http://localhost:9621/query"
    payload = {"query": qa_pair['query'], "mode": mode}
    headers = {"Content-Type": "application/json"}
    if mode == "naive":
        payload["user_prompt"] = user_prompt
    logging.info("Querying in %s mode...", mode)
    async with sem:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
    logging.info("%s mode result saved (%d chars)", mode, len(response.json()["response"]))
    return response.json()["response"]

async def main() -> None:
    user_prompt = """
你是一位电力工程顾问，专注于澳门电力的建设。
请用简短明了的语言回答以下客户问题：
背景信息:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

问题：{{ query }}
"""
    sem = asyncio.Semaphore(5)
    async def handle_one_qa(qa, sem):
        try:
            response = await run_query(type, qa, sem, user_prompt)
            await rate_and_write_output(qa['query'], response, qa['answer'], qa['source'])
        except Exception as e:
            print(f"Error processing QA pair: {e}")
    coros = []
    async with aiofiles.open("/Users/amychan/rag_test_light/LightRAG/outputs/lightrag1/graded.json", "r") as f:
        raw = await f.read()
    json_data = json.loads(raw)
    for _, data in enumerate(json_data):
        coros.append(handle_one_qa(data, sem))
    await asyncio.gather(*coros)

if __name__ == "__main__":
    asyncio.run(main())