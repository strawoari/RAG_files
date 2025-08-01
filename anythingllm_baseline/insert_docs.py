import json
import os
import asyncio
import aiofiles
from dotenv import load_dotenv
import logging
import httpx
import requests
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

load_dotenv()
print(os.getenv("ANYTHING_WORKSPACE"))
print(os.getenv("ANYTHING_API_KEY"))

input_dir_1 = "/Users/amychan/rag_files/data/web_data"
input_dir_2 = "/Users/amychan/rag_files/data/pdf_docs"

async def insert_doc_json(fname, sem) -> None:
    # print(f"Inserting document: {fname}")
    url = 'http://localhost:3001/api/v1/document/raw-text'
    async with sem:
        try:
            async with aiofiles.open(fname, "r") as f:
                raw = await f.read()
            json_data = json.loads(raw)
            text = json_data['page_content']
            metadata = json_data['metadata']
            title = metadata.get("source") or metadata.get("page_label") or fname.stem
            # print(f"Bearer {os.getenv('ANYTHING_API_KEY')}")
            headers = {
                "Authorization": f"Bearer {os.getenv("ANYTHING_API_KEY")}",
                "accept": "application/json",
                "Content-Type": "application/json"
            }
            data = {
                "textContent": text,
                "addToWorkspaces": os.getenv("ANYTHING_WORKSPACE"),
                "metadata": {"title": title}
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=data)
            # print(f"Response: {response.json()}")
            if response.status_code != 200:
                print(response.json())
                raise Exception(f"Failed to insert document: {response.json()}")
        except Exception as e:
            print(f"Error inserting document: {e}")
            raise e

async def main():
    sem = asyncio.Semaphore(6)
    tasks = []
    json_files_1 = [Path(input_dir_1) / f for f in os.listdir(input_dir_1) if f.endswith('.json')]
    json_files_2 = [Path(input_dir_2) / f for f in os.listdir(input_dir_2) if f.endswith('.json')]
    for fname in json_files_1:
        tasks.append(asyncio.create_task(insert_doc_json(fname, sem)))
    for fname in json_files_2:
        tasks.append(asyncio.create_task(insert_doc_json(fname, sem)))
    await asyncio.gather(*tasks)