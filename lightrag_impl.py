import os
import asyncio
import shutil
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
import logging
from openai import AzureOpenAI
from lightrag.kg.shared_storage import initialize_pipeline_status
from load_file import get_docs

# this file is not for running

logging.basicConfig(level=logging.INFO)

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")
EMBEDDING_DIM = os.getenv("EMBEDDING_DIM")

WORKING_DIR = ""
OUTPUT_DIR = ""
MAX_BYTES = int(os.getenv("MAX_BYTES", str(4 * 1024 * 1024)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs")).resolve()

if os.path.exists(WORKING_DIR):
    import shutil
    shutil.rmtree(WORKING_DIR)
os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,  # model = "deployment_name".
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
    )
    return chat_completion.choices[0].message.content


async def embedding_func(texts: list[str]) -> np.ndarray:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    )
    embedding = client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)

    embeddings = [item.embedding for item in embedding.data]
    return np.asarray(embeddings)


async def test_funcs():
    result = await llm_model_func("How are you?")
    print("Response from llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("Response from embedding_func: ", result.shape)
    print("Dimension of embedding: ", result.shape[1])

# asyncio.run(test_funcs())

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim= os.getenv("EMBEDDING_DIM"),
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def write_output(filename_base: str, text: str) -> None:
    """Append *text* to an output file, rolling over when MAX_BYTES reached."""
    idx = 0
    while True:
        file_path = OUTPUT_DIR / f"{filename_base}_{idx}.txt"
        # If writing text would exceed size limit, move to next index
        if file_path.exists() and file_path.stat().st_size + len(text.encode()) > MAX_BYTES:
            idx += 1
            continue
        async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
            await f.write(text + "\n")
        return  # done!

async def run_query(rag: LightRAG, mode: str, query_text: str, user_prompt: str | None = None) -> None:
    """Executes *rag.query* in a thread and writes the result to file."""
    param = QueryParam(mode=mode, user_prompt=user_prompt)
    logging.info("Querying in %s mode...", mode)
    result: str = await asyncio.to_thread(rag.query, query_text, param=param)
    # Persist + log
    await write_output(mode, result)
    logging.info("%s mode result saved (%d chars)", mode, len(result))
    
async def main() -> None:
    rag = await initialize_rag()

    docs = await asyncio.to_thread(get_docs, "/Users/amychan/rag_files/data")
    logging.info("Loaded %d docs; inserting into index…", len(docs))
    await asyncio.to_thread(rag.insert, docs)

    query_text = "澳门电力公司提供哪些服务？"
    
    user_prompt = """
你是一位电力工程顾问，专注于澳门电力的建设。
请用简短明了的语言回答以下客户问题：
背景信息:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

问题：{{ query }}
"""
    tasks = [
        run_query(rag, "naive", query_text, user_prompt),
        run_query(rag, "local", query_text),
        run_query(rag, "global", query_text),
        run_query(rag, "hybrid", query_text),
    ]
    await asyncio.gather(*tasks)

    logging.info("All queries completed. Results are in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()