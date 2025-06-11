import asyncio
import json
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy


class PageData(BaseModel):
    content: str = Field(..., description="对用户有实际参考价值（如具体政策、流程、服务、联系方式、常见问题等）的内容，不包括导航、版权、菜单、广告、无关内容。用Markdown形式返回")
    url: str = Field(..., description="内容来源的URL")

async def main():
    llm_strategy = LLMExtractionStrategy(
        llm_config = LLMConfig(provider="deepseek/deepseek-chat",api_token="sk-9ef8af798e974c38b9d0eb726fa8d463"),
        schema= PageData.model_json_schema(),
        extraction_type="schema",
        instruction="请根据内容返回'url'和'content'。请只返回与用户有实际参考价值的内容（如具体政策、流程、服务、联系方式、常见问题等），完全删除所有导航、版权、菜单、广告、无关内容。不要返回任何无关内容。",
        max_pages = 3
    )

    config = CrawlerRunConfig(
        exclude_external_links=True,
        word_count_threshold=20,
        extraction_strategy=llm_strategy,
        page_timeout=120000,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://www.cem-macau.com/zh/customer-service/downloadarea/application-form-and-declaration", config=config)
        print(result.markdown)
        
    # file_urls = set()
    # for doc in result:
    #     matches = re.findall(r'https?://[^\s\'"]+\.(pdf|txt)', doc.page_content, re.IGNORECASE)
    #     file_urls.update(matches)
    
    # file_docs = []
    # for url in file_urls:
    #     file_docs.append(load_files_from_url(url))

    # docs = result + [item for sublist in file_docs for item in sublist]
    

if __name__ == "__main__":
    asyncio.run(main())