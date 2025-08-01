import json
import os
import asyncio
import re
import aiofiles
from dotenv import load_dotenv
import httpx
import numpy as np
from openai import AsyncAzureOpenAI
import requests
from pathlib import Path
import score_calc
import anythingllm_store
from evaluation import spacy_score

load_dotenv()
print(os.getenv("ANYTHING_WORKSPACE"))
print(os.getenv("ANYTHING_API_KEY"))


AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

OUTPUT_DIR = Path("./outputs/anythingllm").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

embed_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_EMBEDDING_API_VERSION,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
)

async def rate_and_write_output(qa_pair: dict, response: str, contexts: list, index: int) -> None:
    answer_score = spacy_score(response, qa_pair['answer'])
    async with aiofiles.open(OUTPUT_DIR / f"anythingllm_baseline.jsonl", "a", encoding="utf-8") as f:
        await f.write(json.dumps({
            "question": qa_pair['question'],
            "response": response,
            "answer": qa_pair['answer'],
            "answer_score": answer_score,
            "contexts": contexts}, ensure_ascii=False) + "\n")
        

async def run_query(qa_pair: dict, sem: asyncio.Semaphore, session_id: str) -> str:
    url = f"http://localhost:3001/api/v1/workspace/{slug}/chat"
    message = f"你是一个专业的文件分析助手，请根据以下背景信息得出以下问题的答案。用大概两句话回复。\n如果没有相关信息，请回复“目前没有有关资料，请咨询客服人员或查询官网。”\n问题：{qa_pair['question']}"
    payload = {"message": message, 
               "mode": "query", 
               "attachments": [],
               "sessionId": session_id, # TODO: change to divide session by document later 
               }
    headers = {"Authorization": f"Bearer {os.getenv("ANYTHING_API_KEY")}",
                "Accept": "application/json",
                "Content-Type": "application/json"}
    # print(payload)
    async with sem:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
    # print(response.status_code, response.text)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} {response.text}")
    res = response.json()["textResponse"]
    contexts = response.json()["sources"]
    contexts = [{'content': context['text'], 
                 'score': context['score'], 
                 'distance': context['_distance']} for context in contexts]
    print("result saved (%d chars)", len(res))
    return res, contexts

async def main() -> None:
    sem = asyncio.Semaphore(5)
    
    async def handle_one_qa(qa, sem, index, session_id):
        try:
            response, contexts = await run_query(qa, sem, session_id)
            await rate_and_write_output(qa, response, contexts, index)
        except Exception as e:
            raise e
    
    coros = []
    async with aiofiles.open("/Users/amychan/rag_files/lightrag/outputs/v0/annotated.json", "r") as f:
        raw = await f.read()
    json_data = json.loads(raw)
    for _, data in enumerate(json_data):
        q = data['Query']
        a = data['Answer']
        index = data['index']
        coros.append(handle_one_qa({'question': q, 'answer': a}, sem, index, session_id = index))
    await asyncio.gather(*coros)

if __name__ == "__main__":
    asyncio.run(anythingllm_store.main())
    asyncio.run(main())