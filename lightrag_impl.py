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

logging.basicConfig(level=logging.INFO)

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("LIGHT_LLM_ENDPOINT")

AZURE_EMBEDDING_ENDPOINT = os.getenv("LIGHT_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("OPENAI_API_VERSION")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs/lightrag")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pattern = re.compile(
    r"question:\s*(.*?)\n\nanswer:\s*(.*?)\n\nref:\s*(.*?)\n\n",
    re.DOTALL
)

def get_qa_pairs(list_of_qa):
    matches = pattern.findall(list_of_qa)
    lst = []
    for question, answer, source in matches:
        lst.append({
            "question": question.strip(),
            "answer": answer.strip(),
            "source": source.strip()
        })
    return lst

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

embed_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_EMBEDDING_API_VERSION,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
)

async def rate_and_write_output(filename_base: str, query: str, response: str, answer: str, source: str) -> None:
    embedding = await asyncio.to_thread(
        embed_client.embeddings.create,
        model=AZURE_EMBEDDING_DEPLOYMENT,
        input=[response, answer]
    )
    response_embedding, answer_embedding = [item.embedding for item in embedding.data]
    similarity = cosine_similarity(response_embedding, answer_embedding)
    idx = int(similarity * 10)
    file_path = OUTPUT_DIR / f"{filename_base}_{idx}.txt"
    log = f"Query: {query}\nResponse: {response}\nSource: {source}"
    try:
        async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
            await f.write(log + "\n")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
    return  # done!

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

async def insert_doc(doc, sem: asyncio.Semaphore) -> None:
    url = "http://localhost:9621/documents/file"
    headers = {"Content-Type": "application/json"}
    # If doc is a path, open as binary for upload
    file = ('file', open(doc, 'rb'))
    async with sem:
        # Use to_thread for blocking requests
        await asyncio.to_thread(
            httpx.post, url, headers=headers, files={"file": file}
        )
        logging.info("Inserted %s", doc)
    return

async def main() -> None:
    doc_dir = "/Users/amychan/rag_files/pretest_doc_data"
    docs = [os.path.join(doc_dir, f) for f in os.listdir(doc_dir) if f.endswith('.json')]
    logging.info("Loaded %d docs; inserting into index…", len(docs))
    sem = asyncio.Semaphore(6)
    # Insert docs concurrently
    insert_tasks = [asyncio.create_task(insert_doc(doc, sem)) for doc in docs]
    await asyncio.gather(*insert_tasks)

    # Pre-create result files
    for i in range(10):
        file_path = OUTPUT_DIR / f"mode_{i}.txt"
        file_path.touch(exist_ok=True)

    user_prompt = """
你是一位电力工程顾问，专注于澳门电力的建设。
请用简短明了的语言回答以下客户问题：
背景信息:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

问题：{{ query }}
"""
    async def handle_one_qa(qa, sem):
        for mode in ["naive", "local", "global", "hybrid"]:
            try:
                response = await run_query(mode, qa, sem, user_prompt)
                await rate_and_write_output(mode, qa['query'], response, qa['answer'], qa['source'])
            except Exception as e:
                print(f"Error processing QA pair: {e}")
    qa_dir = "/Users/amychan/rag_files/qa_data"
    json_files = [Path(qa_dir) / f for f in os.listdir(qa_dir) if f.endswith('.json')]
    tasks = []
    for fname in json_files:
        try:
            async with aiofiles.open(fname, "r") as f:
                raw = await f.read()
            json_data = json.loads(raw)
            for i, data in enumerate(json_data):
                list_of_qa = get_qa_pairs(data['list_of_qa'][0])
                for qa in list_of_qa:
                    tasks.append(asyncio.create_task(handle_one_qa(qa, sem)))
        except Exception as e:
            print(f"Error reading file {fname}: {e}")
    await asyncio.gather(*tasks)
    logging.info("All queries completed. Results are in %s", OUTPUT_DIR)
    return None

if __name__ == "__main__":
    asyncio.run(main())