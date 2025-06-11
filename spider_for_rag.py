from langchain_community.document_loaders import SpiderLoader
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import asyncio
import re
import os
from dotenv import load_dotenv

load_dotenv()



def main():
    params = {
    "return_format": "text",
    "metadata": True,
    "exclude_urls": ["https://www.cem-macau.com/zh/press-release.*"],
    "depth_limit": 4,
    "limit": 3
    }
    loader = SpiderLoader(
        api_key="sk-092e365c-dd53-412b-96d0-f669be697ca7",
        url="https://www.cem-macau.com/zh/customer-service/downloadarea/application-form-and-declaration",
        mode="crawl",
        params = params
    )

    loaded_docs = loader.load()

    # Add PDF/TXT URLs found on each page to the Document's metadata
    for doc in loaded_docs:
        # Find all PDF/TXT links in the page content
        pdf_urls = re.findall(r'https?://[^\s\'\"]+\.(pdf|txt)', doc.page_content, re.IGNORECASE)
        # Remove image links (e.g., .jpg, .png, .gif, .svg, .webp, etc.)
        # (Not needed here, but if you want to exclude other file types, you can add more logic)
        # Add the list of PDF/TXT URLs to the document's metadata
        doc.metadata["pdf_urls"] = list(set(pdf_urls))
        
    file_urls = set()
    for doc in loaded_docs:
        matches = re.findall(r'https?://[^\s\'"]+\.(pdf|txt)', doc.page_content, re.IGNORECASE)
        file_urls.update(matches)

    file_docs = []
    for url in file_urls:
        file_docs.append(load_file_from_url(url))

    docs = loaded_docs + [item for sublist in file_docs for item in sublist]

    llm = AzureChatOpenAI(model= os.getenv("AZURE_OPENAI_DEPLOYMENT")),
    prompt = PromptTemplate.from_template("""
    请判断以下网页内容是否对用户有实际参考价值（如具体政策、流程、服务、联系方式、常见问题等），如果只是导航、版权、菜单、广告、无关内容，请删除无关内容。
    内容：
    {text}
    URL: {url}
    返回且只返回有关内容。
    """)
    filtered_docs = asyncio.run(process_documents(llm, prompt, docs))
    f = open("filtered_docs.txt", "w")
    for doc in filtered_docs:
        f.write(doc.page_content)
    f.close()

# Load Documents from the URLs     
def load_file_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        suffix = ".pdf" if url.lower().endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(response.content)
            f.flush()
            if suffix == ".pdf":
                loader = PyPDFLoader(f.name)
            else:
                loader = TextLoader(f.name, autodetect_encoding=True)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
            return docs
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return []
    
async def process_documents(llm, prompt, documents):
    async def filter_doc(doc: Document):
        formatted_prompt = prompt.format(text=doc.page_content, url=doc.metadata.get("url", ""))
        response = await llm.ainvoke(formatted_prompt)
        content = response.content.strip()
        if content:
            return Document(page_content=content, metadata=doc.metadata)
        return None

    tasks = [filter_doc(doc) for doc in documents]
    filtered = await asyncio.gather(*tasks)
    return [doc for doc in filtered if doc]

if __name__ == "__main__":
    asyncio.run(main())
