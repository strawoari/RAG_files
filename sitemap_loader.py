from langchain_community.document_loaders.sitemap import SitemapLoader
from bs4 import BeautifulSoup
from openai import AzureOpenAI
import os
import json
import re

def call_spider(given_url, exclude_urls):
    sitemap_loader = SitemapLoader(
        web_path=given_url,
        filter_urls=exclude_urls,
        continue_on_failure=True
    )
    return sitemap_loader.load()  # ← YOU FORGOT TO RETURN THIS

def html_doc(doc):
    html_content = doc.page_content
    url = doc.metadata.get('source') or doc.metadata.get('loc')
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()

    text = soup.get_text(separator=' ', strip=True)
    text = ' '.join(text.split())

    if text:
        return {'text': text, 'url': url}
    return None

def clean_content(text):
    text = re.sub(r"https?://[^\s'\")\]]+?\.svg", "", text)
    text = re.sub(r'!\[\]\((?![^)]*\.pdf)[^)]*\)', '', text)
    text = re.sub(r'data:image/[^)]*', '', text)
    return text

def load_docs(given_url, exclude_urls):
    loaded_docs = call_spider(given_url, exclude_urls)
    if loaded_docs is None:
        loaded_docs = []

    print("Loaded documents from the spider:", len(loaded_docs))

    cleaned_docs = []
    for doc in loaded_docs:
        parsed = html_doc(doc)
        if parsed:
            parsed["text"] = clean_content(parsed["text"])
            cleaned_docs.append(parsed)
        else:
            print("Failed to clean document, skipping.")

    print("First cleaned document:\n", cleaned_docs[0])

    client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    )

    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not model_name:
        raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT_NAME in environment variables")

    infos = []
    for doc in cleaned_docs:
        text = doc['text']
        url = doc['url']
        string = ""
        start = 0
        chunk_size = 100000  # adjust if needed

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            prompt = f"""请判断以下网页内容是否对用户有实际参考价值（如具体政策、流程、服务、联系方式、常见问题等），如果只是导航、版权、菜单、广告、重复的标点符号，无关内容，请删除无关内容。
内容：
{chunk}
URL: {url}
返回且只返回有关内容。"""

            messages = [
                {"role": "system", "content": "你是一个内容筛选助手。"},
                {"role": "user", "content": prompt}
            ]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                top_p=1,
                n=1,
            )
            content = response.choices[0].message.content.strip()
            string += content
            start = end

        infos.append({'url': url, 'content': string})

    with open("contents.json", 'w', encoding='utf-8') as f:
        json.dump(infos, f, ensure_ascii=False, indent=4)

    print("First processed entry:\n", infos[0])

if __name__ == "__main__":
    load_docs(
        "https://www.cem-macau.com/zh/customer-service/downloadarea/application-form-and-declaration",
        ["https://www.cem-macau.com/zh/press-release.*"]
    )