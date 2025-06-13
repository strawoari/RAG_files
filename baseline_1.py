from load_file import get_docs
import os, json, re, asyncio, numpy as np
from functools import partial
import json
import numpy as np
from openai import AsyncAzureOpenAI 
from basic_rag import BasicRAG

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

def cosine_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

pattern = re.compile(
    r"question:\s*(.*?)\n\nanswer:\s*(.*?)\n\nref:\s*(.*?)\n\n",
    re.DOTALL
  )

def get_qa_pairs(list_of_qa):
  matches = pattern.findall(list_of_qa)
  lst = []
  for question, answer, source in matches:
    lst.append({
      "question": question.strip() ,
      "answer": answer.strip(),
      "source": source.strip()
    })
  return lst

embed_client = AsyncAzureOpenAI(
          api_key=AZURE_OPENAI_API_KEY,
          api_version=AZURE_EMBEDDING_API_VERSION,
          azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    )

async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))

async def async_client_query(rag, query: str) -> str:
    """Calls BasicRAG.client_query in a worker thread."""
    resp = await run_in_thread(rag.client_query, query)
    # rag.client_query returns list-of‑strings per your code
    if isinstance(resp, (list, tuple)):
      return resp[0]
    else:
      return resp

async def get_and_rate_answer(rag, qa_pair, sem: asyncio.Semaphore):
    question = qa_pair['question']
    answer = qa_pair['answer']
    source = qa_pair['source']
    async with sem:                         # throttle concurrent calls
      response = await async_client_query(rag, question)
      embedding = embed_client.embeddings.create(
        model=AZURE_EMBEDDING_DEPLOYMENT, 
        input=[response, answer])
    response_embedding, answer_embedding = [item.embedding for item in embedding.data]
    similarity = cosine_similarity(response_embedding, answer_embedding)
    return {
        "question": question,
        "response": response,
        "similarity": similarity,
        "answer": answer,
        "source": source,
    }

import aiofiles

async def gather_all(rag, input_dir: str, output_dir: str, max_concurrency: int = 5, max_byte: int = 100 * 1024 * 1024):
    os.makedirs(output_dir, exist_ok=True)
    sem  = asyncio.Semaphore(max_concurrency)
    tasks = []
    current_file_index = 0
    current_file_size = 0
    current_file_path = os.path.join(output_dir, f"results_{current_file_index}.jsonl")
    file_handle = await aiofiles.open(current_file_path, mode="w")
    
    async def write_to_file(result):
        nonlocal current_file_index, current_file_size, file_handle

        line = json.dumps(result, ensure_ascii=False) + "\n"
        line_size = len(line.encode("utf-8"))

        # If this line would make the file too large, roll over to new file
        if current_file_size + line_size > max_bytes_per_file:
            await file_handle.close()
            current_file_index += 1
            current_file_size = 0
            current_file_path = os.path.join(output_dir, f"results_{current_file_index}.jsonl")
            file_handle = await aiofiles.open(current_file_path, mode="w")
        await file_handle.write(line)
        current_file_size += line_size
        return None
        
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    for fname in json_files:
        async with aiofiles.open(fname, "r") as f:
            raw = await f.read()
        json = json.loads(raw)
        for i, data in enumerate(json):
            list_of_qa = data['list_of_qa'][0]
            for qa in list_of_qa:
                async def handle_one_qa(qa, source):
                    result = await get_and_rate_answer(rag, qa, sem)
                    result["file_index"] = source
                    await write_to_file(result)
                tasks.append(asyncio.create_task(handle_one_qa(qa=qa, source=f"{fname}_{i}")))
            
    await asyncio.gather(*tasks)
    await file_handle.close()
    return None

if __name__ == "__main__":
    template = """
    你是一位电力工程顾问，专注于澳门电力的建设。
    请用简短明了的语言回答以下客户问题：
    背景信息:
    {% for document in documents %}
    {{ document.content }}
    {% endfor %}

    问题：{{ query }}
    """
    rag = BasicRAG(template)
    # comment out if dataset rag already created document store!
    docs = get_docs('/Users/amychan/rag_files/data')
    # comment out if dataset rag already created document store!
    asyncio.run(run_in_thread(rag.store_documents, rag.index_pipeline, docs))
    print("Document store created, now running QA pairs")
    asyncio.run(
        gather_all(rag, "/Users/amychan/rag_files/data", max_concurrency=5)
    )
    