from langchain_community.document_loaders.sitemap import SitemapLoader
from bs4 import BeautifulSoup
from openai import AzureOpenAI

sitemap_loader = SitemapLoader(
    web_path="https://www.cem-macau.com/sitemap.xml",
    filter_urls=["https://www.cem-macau.com/zh(?!/press-release).*"],
    continue_on_failure=True  # Added this line to continue loading on failure
)
docs = sitemap_loader.load()

text_docs = []
for doc in docs:
    if hasattr(doc, 'page_content'):
        html_content = doc.page_content
        url = getattr(doc, 'metadata', {}).get('source', None) or getattr(doc, 'metadata', {}).get('loc', None)
    else:
        html_content = doc
        url = None

    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    # Get text and clean it
    text = soup.get_text(separator=' ', strip=True)
    # Remove excessive whitespace
    text = ' '.join(text.split())
    if text:  # Only add non-empty texts
        text_docs.append({'text': text, 'url': url})

docs = text_docs
print(docs[0])
client = AzureOpenAI(
        api_key= os.getenv('AZURE_OPENAI_API_KEY'),
        api_version= LLM_ENDPOINT,
        azure_endpoint= LLM_DEPLOYMENT,
    )
prompt = f"""
        请判断以下网页内容是否对用户有实际参考价值（如具体政策、流程、服务、联系方式、常见问题等），如果只是导航、版权、菜单、广告、无关内容，请删除无关内容。\n内容：{text}\nURL: {url}\n返回且只返回有关内容。"
        """
infos = []
for i in docs:
  messages = [{"role": "system", "content": "你是一个内容筛选助手。"}]
  text = i['text']
  url = i['url']
  messages.append({"role": "user", "content": f"""
        请判断以下网页内容是否对用户有实际参考价值（如具体政策、流程、服务、联系方式、常见问题等），如果只是导航、版权、菜单、广告、无关内容，请删除无关内容。\n内容：{text}\nURL: {url}\n返回且只返回有关内容。"
        """})
  chat_completion = client.chat.completions.create(
      model=LLM_DEPLOYMENT,  # model = "deployment_name".
      messages=messages,
      temperature=kwargs.get("temperature", 0),
      top_p=kwargs.get("top_p", 1),
      n=kwargs.get("n", 1),
  )
  infos.append({'url': url, 'content': chat_completion.choices[0].message.content})
print(infos[0])
f = open('text.txt', 'w')
f.write(str(infos))
f.close()