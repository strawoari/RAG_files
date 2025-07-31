import os
import asyncio
from dotenv import load_dotenv
import logging
import httpx
from pathlib import Path

logging.basicConfig(level=logging.INFO)

load_dotenv()


async def insert_docs(docs) -> None:
    url = "http://localhost:1024/documents/file_batch"
    print(docs[0])
    files = [("files",(os.path.basename(doc), open(doc, 'rb'), 'application/json')) for doc in docs]
    # Use to_thread for blocking requests
    await asyncio.to_thread(
        httpx.post, url, files = files
    )
    logging.info("Inserted %s docs", len(docs))
    return

async def main() -> None:
    doc_dir_1 = "/Users/amychan/rag_files/data/pdf_docs"
    docs = [Path(doc_dir_1) / f for f in os.listdir(doc_dir_1) if f.endswith('.json')]
    await insert_docs(docs)
    logging.info("Loaded %d docs; inserting into index…", len(docs))
    doc_dir_2 = "/Users/amychan/rag_files/data/web_data"
    docs_2 = [Path(doc_dir_2) / f for f in os.listdir(doc_dir_2) if f.endswith('.json')]
    await insert_docs(docs_2)
    logging.info("Loaded %d docs; inserting into index…", len(docs_2))
    # sem = asyncio.Semaphore(6)
    return
    
if __name__ == "__main__":
    asyncio.run(main())
    
