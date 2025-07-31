import asyncio
import hashlib
import json
import os
import aiofiles
import aiohttp
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import logging
from langchain.document_loaders import UnstructuredPDFLoader

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),                  # Terminal
        logging.FileHandler("output.log", mode='w')  # Log file
    ]
)

logging.info("This shows in terminal AND in output.log")

# async def load_file_from_url(save_dir, session, url, sem: asyncio.Semaphore):
#     async with sem:
#         for attempt in range(3):
#             try:
#                 async with session.get(url, timeout=30) as response:
#                     response.raise_for_status()
#                     content = await response.read()
#                     # print(f"Request for {url} printed {response.status}")
#                     content_type = response.headers.get('Content-Type', '')
#                     if "application/pdf" not in content_type:
#                         raise Exception(f"Unexpected content-type: {content_type}")

#                     content = await response.read()
#                     if not content:
#                         raise Exception("Empty content")
#                     filename = url.split("/")[-1].split("?")[0]  # Get the file name from URL
#                     save_path = os.path.join(save_dir, filename)
#                     async with aiofiles.open(save_path, 'wb') as f:
#                         await f.write(content)
#                     print(f"Downloaded {url} to {save_path} and saved to {save_path}")
                    
#                     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
#                         f.write(content)
#                         f.flush()
#                         loader = PyPDFLoader(f.name)
#                         docs = loader.load()
#                         # print(f"Loaded {len(docs)} documents from {url}")
#                         for doc in docs:
#                             doc.metadata["source"] = url
#                         return docs
#             except Exception as e:
#                 if attempt < 2:
#                     # print(f"Error loading {url}: {e}, retrying...")
#                     await asyncio.sleep(3) # Wait before retrying
#                 else:
#                     print(f"Error loading {url}: {e}, skipping...")
#                     return []

def save_docs_in_directory(filtered_docs, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for idx, doc in enumerate(filtered_docs):
        # Use a hash of the content for uniqueness, or just use idx
        filename = doc.metadata.get("source").split("/")[-1].split("?")[0]
        # doc_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        filename = f"doc_{idx}_{filename}.json"
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }, f, ensure_ascii=False, indent=4)
            
async def main():
    # async with aiohttp.ClientSession() as session:
    #     sem = asyncio.Semaphore(5)
    #     file_urls = []
    #     with open("data/file_urls.txt", "r") as f:
    #         for line in f:
    #             file_urls.append(line.strip())
    #     print(f"Loaded {len(file_urls)} file URLs")
    #     tasks = [load_file_from_url("./data/pdfs", session, url, sem) for url in file_urls]
    #     result = await asyncio.gather(*tasks)
    #     result = [doc for docs in result if docs for doc in docs]
    #     save_docs_in_directory(result, data_dir="./data/pdf_docs")
    pdf_path = "./data/pdfs"
    for file in os.listdir(pdf_path):
        loader = UnstructuredPDFLoader(os.path.join(pdf_path, file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file
        save_docs_in_directory(docs, data_dir="./data/pdf_docs")

if __name__ == "__main__":
    asyncio.run(main())
    
