import os
from langchain_community.document_loaders import SpiderLoader
from dotenv import load_dotenv
load_dotenv()

def call_spider(given_url, exclude_urls):
    params = {
    "return_format": "markdown",
    "return_json_data": False,
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

if __name__ == "__main__":
    # Example usage
    given_url = "https://example.com"
    exclude_urls = ["https://example.com/exclude"]
    
    loaded_docs = call_spider(
        "https://www.cem-macau.com/zh/customer-service/downloadarea/application-form-and-declaration",
        ["https://www.cem-macau.com/zh/press-release.*"])
    print(loaded_docs)