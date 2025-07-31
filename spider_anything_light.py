from langchain_community.document_loaders import SpiderLoader
from langchain_core.documents import Document
from openai import AzureOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import aiohttp
import aiofiles
import tempfile
import json
import asyncio
import re
import os
from dotenv import load_dotenv
import hashlib

load_dotenv()

def call_spider(given_url, exclude_urls):
    params = {
    "return_format": "markdown",
    "return_json_data": True,
    "metadata": True,
    "exclude_urls": exclude_urls,
    "depth_limit": 4,
    "limit": 1
    }
    loader = SpiderLoader(
        api_key= os.getenv("SPIDER_API_KEY"),
        url=given_url,
        mode="crawl",
        params = params
    )
    return loader.load()

async def download_pdf(session, url, save_dir):
    filename = url.split("/")[-1].split("?")[0]  # Get the file name from URL
    save_path = os.path.join(os.basename(save_dir), filename)
    try:
        async with session.get(url, timeout=20) as response:
            response.raise_for_status()
            async with aiofiles.open(save_path, 'wb') as f:
                await f.write(await response.read())
        print(f"Downloaded {url} to {save_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

async def record_file_docs(loaded_docs):
    file_urls = set()
    for doc in loaded_docs:
        urls = re.findall(r"https?://[^\s'\")\]]+?\.pdf", doc.page_content) # wrap the reg into a whole group
        file_urls.update(urls)
    
    async with aiohttp.ClientSession() as session:
        tasks = [download_pdf(session, url, 
                              "/Users/amychan/rag_test_light/pretest_doc_data/anythingllm"
                              ) for url in file_urls]
        await asyncio.gather(*tasks)
    
def clean_content(text):
    text = re.sub(r"https?://[^\s'\")\]]+?\.svg", "", text)
    text = re.sub(r'!\[\]\((?![^)]*\.pdf)[^)]*\)', '', text)# Remove markdown image links that are not PDFs
    text = re.sub(r'data:image/[^)]*', '', text)# Remove inline base64 images
    return text

async def load_docs(given_url, exclude_urls):
    loaded_docs = call_spider(given_url, exclude_urls)
    if loaded_docs is None:
        loaded_docs = []
    print("Loaded documents from the spider:" + str(len(loaded_docs)))
    await record_file_docs(loaded_docs)
    cleaned_docs = [
        Document(page_content= clean_content(doc.page_content), metadata=doc.metadata)
        for doc in loaded_docs
    ]
    print("First cleaned document:\n" + str(cleaned_docs[0].page_content[:30]))
    if file_docs is None or file_docs == []:
        print("No file docs were loaded successfully")
        file_docs = []
    
    client = AzureOpenAI(
        api_key= os.getenv("AZURE_OPENAI_API_KEY"),
        api_version= os.getenv("OPENAI_API_VERSION"),
        azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    prompt = PromptTemplate.from_template("""
    请判断以下原文是否对用户有实际参考价值（如具体政策、流程、服务、联系方式、常见问题等），摘抄有关内容。
    尊守以下三点：
    1.删除所有导航、版权、菜单、广告、重复的标点符号，等无关内容。
    2.如果找到有关内容，按原文抄写。
    3.不要总结内容。有必要时可以加入不超过1句的解释。
    参考原文关键词: {keywords}
    原文：
    {text}
    """)
    print('start filtering')

    async def process_documents(documents, max_concurrent=6):
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def filter_doc(doc: Document):
            async with semaphore:
                string = ""
                start = 0
                raw_kw = doc.metadata.get("keywords", "")
                keywords = ""
                if isinstance(raw_kw, (list, tuple)):          # flatten list → comma‑sep string
                    keywords = ", ".join(map(str, raw_kw))
                else:
                    keywords = str(raw_kw)
                print(keywords)
                while start < len(doc.page_content):
                    end = min(start + 100000, len(doc.page_content))
                    chunk = doc.page_content[start:end]
                    last_newline = chunk.rfind('\n')
                    if last_newline != -1 and end != len(doc.page_content):
                        split_point = start + last_newline + 1
                    else:
                        split_point = end
                    formatted_prompt = prompt.format(
                        text=doc.page_content[start:split_point],
                        keywords = keywords
                    )
                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        messages=[{
                            "role": "user",
                            "content": formatted_prompt
                        }],
                        model=os.getenv("AZURE_OPENAI_DEPLOYMENT")
                    )
                    content = response.choices[0].message.content.strip()
                    if content:
                        string += (content)
                    start = split_point
                doc.metadata["page_label"] = keywords
                results.append(Document(page_content=string, metadata=doc.metadata))

        await asyncio.gather(*(filter_doc(doc) for doc in documents))
        return results
    # 5. Run the filtering
    filtered_docs = await process_documents(cleaned_docs)

    save_docs_in_directory(filtered_docs, data_dir="pretest_doc_data/anythingllm")
    return None

def save_docs_in_directory(filtered_docs, data_dir = "doc_data/anythingllm"):
    os.makedirs(data_dir, exist_ok=True)
    for idx, doc in enumerate(filtered_docs):
        # Use a hash of the content for uniqueness, or just use idx
        doc_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        filename = f"doc_{idx}_{doc_hash}.json"
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    asyncio.run(load_docs(
        "https://www.cem-macau.com/zh/customer-service/downloadarea/application-form-and-declaration",
        ["https://www.cem-macau.com/zh/press-release.*"]
    ))