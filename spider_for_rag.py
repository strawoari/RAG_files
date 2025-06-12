from langchain_community.document_loaders import SpiderLoader
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import json
import asyncio
import re
import os
from dotenv import load_dotenv
import requests
import hashlib


load_dotenv()

# Load Documents from the URLs     
def load_file_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
            f.flush()
            print(f.name)
            # print(open(f.name, "r").read())
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
            return docs
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return []

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
    
def get_file_docs(loaded_docs):
    file_urls = set()
    for doc in loaded_docs:
        urls = re.findall(r"https?://[^\s'\")\]]+?\.pdf", doc.page_content) # wrap the reg into a whole group
        file_urls.update(urls)
    
    file_docs = []
    for url in file_urls:
        print(f"\nTrying to load URL: {url}")
        docs = load_file_from_url(url)
        if docs:
            print(f"Successfully loaded {url}")
            file_docs.extend(docs)
        else:
            print(f"Failed to load {url}")
    
    if file_docs:
        print("\nFirst file doc content:")
        print(file_docs[0])
    else:
        print("\nNo file docs were loaded successfully")
    
def load_docs(given_url, exclude_urls):
    loaded_docs = call_spider(given_url, exclude_urls)
    # print(loaded_docs[0].page_content)
    file_docs = []
    if loaded_docs is None:
        print("No docs were loaded successfully")
        loaded_docs = []
    else:
        file_docs = get_file_docs(loaded_docs)
    if file_docs is None or file_docs == []:
        print("No file docs were loaded successfully")
        file_docs = []
    docs = loaded_docs + file_docs
    print(docs[0])
    
    llm = AzureChatOpenAI(model= os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    prompt = PromptTemplate.from_template("""
    请判断以下网页内容是否对用户有实际参考价值（如具体政策、流程、服务、联系方式、常见问题等），如果只是导航、版权、菜单、广告、无关内容，请删除无关内容。
    内容：
    {text}
    URL: {url}
    返回且只返回有关内容。
    """)
    print('start filtering')

    async def process_documents(documents):
        results = []

        async def filter_doc(doc: Document):
            string = ""
            start = 0
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
                    url=doc.metadata.get("source", "")
                )
                response = await llm.ainvoke(formatted_prompt)
                content = response.content.strip()
                if content:
                    string += content
                start = split_point
            results.append(Document(page_content=string, metadata=doc.metadata))

        # Actually process all documents
        await asyncio.gather(*(filter_doc(doc) for doc in documents))
        return results
    # 5. Run the filtering
    filtered_docs = asyncio.run(process_documents(docs))

    save_docs_in_directory(filtered_docs, data_dir="data")
    return None

def save_docs_in_directory(filtered_docs, data_dir="data"):
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
    load_docs(
        "https://www.cem-macau.com/zh/",
        ["https://www.cem-macau.com/zh/press-release.*"]
    )