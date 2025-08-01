import asyncio
import tempfile
import aiohttp
from langchain_community.document_loaders.sitemap import SitemapLoader
from bs4 import BeautifulSoup
from openai import AzureOpenAI
import os
import json
import re
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import hashlib
import tempfile
import aiofiles
from urllib.parse import urljoin
from dataclasses import dataclass

# Use a simple document wrapper
@dataclass
class Document:
    url: str
    page_content: str

async def fetch_html(session, url):
    for attempt in range(3):
        try:
            async with session.get(url, timeout=15) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return ""

async def load_docs_from_sitemap(sitemap_url):
    async with aiohttp.ClientSession() as session:
        sitemap_xml = await fetch_html(session, sitemap_url)
        soup = BeautifulSoup(sitemap_xml, 'xml')
        urls = [loc.text for loc in soup.find_all('loc') 
                if not "press-release" in loc.text and 
                not "/event/" in loc.text and 
                not "/en/" in loc.text and 
                not "/pt/" in loc.text]

        print(f"Found {len(urls)} URLs in sitemap")

        semaphore = asyncio.Semaphore(10)
        loaded_docs = []

        async def fetch_and_wrap(url):
            async with semaphore:
                html = await fetch_html(session, url)
                if html:
                    loaded_docs.append(Document(url=url, page_content=html))

        await asyncio.gather(*(fetch_and_wrap(url) for url in urls))
        return loaded_docs

async def call_spider(given_url, pattern):
    sitemap_loader = SitemapLoader(
        web_path=given_url,
        filter_urls=[pattern],
        continue_on_failure=True
    )
    # Run the blocking load() in a thread to avoid event loop issues
    result = await asyncio.to_thread(sitemap_loader.load)
    return result

def html_doc(text):
    html_content = text
    # url = doc.metadata.get('source') or doc.metadata.get('url')
    soup = BeautifulSoup(html_content, 'html.parser')

    # # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()

    text = soup.get_text(separator=' ', strip=True)
    text = ' '.join(text.split())
    return text

def clean_content(text):
    text = re.sub(r"https?://[^\s'\")\]]+?\.svg", "", text)
    # text = re.sub(r'!\[\]\((?![^)]*\.pdf)[^)]*\)', '', text)
    text = re.sub(r'data:image/[^)]*', '', text)
    return text


async def record_file_docs(loaded_docs):
    file_urls = set()
    for doc in loaded_docs:
        cleaned_content = clean_content(doc.page_content)
        urls = re.findall(r"https?://[^\s'\")\]]+?\.pdf", cleaned_content) # wrap the reg into a whole group
        #print(f"doc: {doc}")
        #print(f"urls: {urls}")
        file_urls.update(urls)
    file_urls = list(file_urls)
    log = ""
    for url in file_urls:
        log += url + "\n"
    with open("data/file_urls.txt", "w") as f:
        f.write(log)
    # print(f"Found {len(file_urls)} pdfs to download")
    # async with aiohttp.ClientSession() as session:
    #     os.makedirs("./data/pdfs", exist_ok=True)
    #     tasks = [download_pdf(session, url, 
    #                           "./data/pdfs"
    #                           ) for url in file_urls]
    #     await asyncio.gather(*tasks)

# async def get_file_docs(loaded_docs):
#     file_urls = set()
#     for doc in loaded_docs:
#         print(f"doc: {doc}")
#         urls = re.findall(r"https?://[^\s'\")\]]+?\.pdf", doc.page_content) # wrap the reg into a whole group
#         file_urls.update(urls)
    
#     file_docs = []
#     sem = asyncio.Semaphore(5)
#     async with aiohttp.ClientSession() as session:
#         tasks = []
#         tasks = [load_file_from_url(session, url, sem) for url in file_urls]
#         result = await asyncio.gather(*tasks)
    
#     for docs in result:
#         if docs:
#             file_docs.extend(docs)
#     return file_docs

async def load_docs(sitemap_url, pattern):
    loaded_docs = await load_docs_from_sitemap(sitemap_url)
    # loaded_docs = await call_spider(sitemap_url, pattern)
    # if loaded_docs is None:
    #     loaded_docs = []
    # print("Loaded documents from the spider:" + str(len(loaded_docs)))
    # loaded_docs = [doc for doc in loaded_docs if "press-release" and "/event/" not in doc.metadata.get('source')]
    # print(f"doc: {loaded_docs[0]}")
    await record_file_docs(loaded_docs)
    # cleaned_docs = []
    # for doc in loaded_docs:
    #     print(doc.metadata.get('source'))
    #     content = html_doc(clean_content(doc.page_content))
    #     if content != '':
    #         cleaned_docs.append(Document(page_content=content, metadata=doc.metadata))
    #     else:
    #         print(f"doc {doc.metadata.get('source')} got no text after cleaning")
    # print("First cleaned document:\n" + str(cleaned_docs[0].page_content[:30]))
    # # if file_docs is None or file_docs == []:
    # #     print("No file docs were loaded successfully")
    # #     file_docs = []
    # docs = cleaned_docs
    
    # client = AzureOpenAI(
    #     api_key= os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_version= os.getenv("OPENAI_API_VERSION"),
    #     azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
    # )
    # prompt = PromptTemplate.from_template("""
    # 请判断以下原文是否对用户有实际参考价值（如具体政策、流程、服务、联系方式、常见问题等），摘抄有关内容。
    # 尊守以下三点：
    # 1.删除所有导航、版权、菜单、广告、重复的标点符号，等无关内容。
    # 2.如果找到有关内容，按原文抄写。
    # 3.不要总结内容。有必要时可以加入不超过1句的解释。
    # 原文：
    # {text}
    # """)
    # print('start filtering')

    # async def process_documents(documents, max_concurrent=6):
    #     results = []
    #     semaphore = asyncio.Semaphore(max_concurrent)
        
    #     async def filter_doc(doc: Document):
    #         async with semaphore:
    #             string = ""
    #             start = 0
    #             # raw_kw = doc.metadata.get("keywords", "")
    #             # keywords = ""
    #             # if isinstance(raw_kw, (list, tuple)):          # flatten list → comma‑sep string
    #             #     keywords = ", ".join(map(str, raw_kw))
    #             # else:
    #             #     keywords = str(raw_kw)
    #             # print(keywords)
    #             while start < len(doc.page_content):
    #                 end = min(start + 100000, len(doc.page_content))
    #                 chunk = doc.page_content[start:end]
    #                 last_newline = chunk.rfind('\n')
    #                 if last_newline != -1 and end != len(doc.page_content):
    #                     split_point = start + last_newline + 1
    #                 else:
    #                     split_point = end
    #                 formatted_prompt = prompt.format(
    #                     text=doc.page_content[start:split_point],
    #                     # keywords = keywords
    #                 )
    #                 response = await asyncio.to_thread(
    #                     client.chat.completions.create,
    #                     messages=[{
    #                         "role": "user",
    #                         "content": formatted_prompt
    #                     }],
    #                     model=os.getenv("AZURE_OPENAI_DEPLOYMENT")
    #                 )
    #                 content = response.choices[0].message.content.strip()
    #                 if content:
    #                     string += (content)
    #                 start = split_point
    #             # doc.metadata["page_label"] = keywords
    #             results.append(Document(page_content=string, metadata=doc.metadata))

    #     await asyncio.gather(*(filter_doc(doc) for doc in documents))
    #     return results
    # # 5. Run the filtering
    # filtered_docs = await process_documents(docs)

    # save_docs_in_directory(filtered_docs, data_dir="data")
    # return None

def save_docs_in_directory(filtered_docs, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for idx, doc in enumerate(filtered_docs):
        # Use a hash of the content for uniqueness, or just use idx
        doc_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        filename = f"doc_{idx}_{doc.metadata.get('source')}.json"
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    asyncio.run(load_docs(
        "https://www.cem-macau.com/sitemap.xml",
        # "https://www.cem-macau.com/zh/customer-service/downloadarea/pamphlet",
        "^https://www\.cem-macau\.com/zh(?:/(?!press_release).*)?$"
    ))