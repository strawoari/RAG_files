import asyncio
import json
import os
import unicodedata
from langchain_community.document_loaders import SpiderLoader
from langchain_core.documents import Document
import os
import asyncpraw
from aiohttp import ClientSession, ClientTimeout
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
import asyncprawcore
import queue
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import requests
import re

load_dotenv()

def write_output(question_set, website, mode = "json"):
    if not os.path.exists("outputs/web_datasets"):
        os.makedirs("outputs/web_datasets")
    if mode == "txt":
        with open(f"outputs/web_datasets/{website}_question_set.{mode}", "w") as f:
            f.write(question_set)
    else:
        with open(f"outputs/web_datasets/{website}_question_set.{mode}", "w", encoding="utf-8") as f:
            json.dump(question_set, f, indent=4)

async def get_full_page_html(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until='networkidle')  # wait for JS-heavy content
        content = await page.content()  # full rendered HTML
        await browser.close()
        return content

async def get_us_faq():
    urls = ["https://www.ladwp.com/support/billingaccount-issues",
                "https://www.ladwp.com/support/payments",
                "https://www.ladwp.com/support/website-issues",
                "https://www.ladwp.com/support/miscellaneous",
                "https://www.ladwp.com/support/service-requests",
                "https://www.constellation.com/energy-101/faqs/home-energy-faqs.html"
                ]
    all_lst = ""
    pattern = re.compile(
        r"Q:\s*(.*?[?.!])(?:\s+A:|$)", 
        re.DOTALL
    )
    async def get_questions(url):
        nonlocal all_lst
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        # print(text)
        start_marker = "Return to Support page"
        end_marker = "EnglishEspañolKorean"
        start = text.find(start_marker)
        end = text.find(end_marker)
        text = text[start + len(start_marker):end]
        print(text[:10] + ", " + text[-10:])
        
        lst = ""
        questions = pattern.findall(text)
        # print(questions[0])
        for i in questions:
            lst += i.strip() + "\n"
        all_lst += f"**{url}\n{lst}\n"
    
    tasks = [asyncio.create_task(get_questions(url)) for url in urls]
    await asyncio.gather(*tasks)
    # Write the output to a file
    write_output(all_lst, "ladwp", "txt")
    return
    
async def get_questions_from_hk_elec():
    keywords = ["account-information", 
                "apply-for-electricity-supply-and-other-services",
                "billing-and-payment",
                "deposit",
                "electricity-consumption",
                "hkelectric-app"]
    url = "https://www.hkelectric.com/zh/customer-services/faqs"
    all_lst = ""
    pattern = re.compile(
        r"[^？。！]*[？\?]", 
        re.DOTALL
    )
    async def get_questions(url, keyword):
        nonlocal all_lst
        html = await get_full_page_html(url + f"#{keyword}")
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        start_marker = "賬戶資料申請電力及其他服務賬單及繳費按金用電知多少港燈應用程式賬戶資料"
        end_marker = "用電知多少賬戶事宜緊急事故聯絡平等機會加入港燈網頁指南聯絡我們"
        start = text.find(start_marker)
        end = text.find(end_marker)
        text = text[start + len(start_marker):end]
        print(text[:10] + ", " + text[-10:])
        
        lst = ""
        questions = pattern.findall(text)
        print(questions[0])
        for i in questions:
            lst += i.strip() + "\n"
        all_lst += f"**{url}#{keyword}**\n{lst}\n"
    
    tasks = [asyncio.create_task(get_questions(url, keyword)) for keyword in keywords]
    await asyncio.gather(*tasks)
    write_output(all_lst, "hk_elec", "txt")
    return

def call_spider(given_url):
    params = {
    "return_format": "text",
    "return_json_data": True,
    "metadata": False,
    }
    loader = SpiderLoader(
        api_key= os.getenv("SPIDER_API_KEY"),
        url=given_url,
        mode="scrape",
        params = params
    )
    return loader.load()

async def get_cem_faq():
    urls = ["https://www.cem-macau.com/zh/faq/5", 
            "https://www.cem-macau.com/zh/faq/8",
            "https://www.cem-macau.com/zh/faq/10",
            "https://www.cem-macau.com/zh/faq/6",
            "https://www.cem-macau.com/zh/faq/4",
            "https://www.cem-macau.com/zh/faq/7",
            "https://www.cem-macau.com/zh/faq/11"]
    tasks = []
    pattern = pattern = re.compile(
        r"([^\n？\?]+?[？\?])"      # Q: everything up to ? / ？
        r"([\s\S]+?)"            # A: as much as possible …
        r"(?=(?:[^\n？\?]+?[？\?])|$)",# look‑ahead – stop right before the next question or EOF
        re.DOTALL
    )
    all_text = ""
    async def scrape_faq(url):
        nonlocal all_text
        doc = call_spider(url)
        text = doc[0].page_content
        start_marker = "返回主目錄"
        end_marker = "客戶服務客戶服務賬單資訊下載區服務承諾"
        start = text.find(start_marker)
        print(start)
        end = text.find(end_marker)
        print(end)
        text = text[start + len(start_marker):end]
        print(text[:10] + ", " + text[-10:])
        qa_pairs = pattern.findall(text)
        # print(qa_pairs)
        flattened = []
        for q, a in qa_pairs:
            if q.strip().endswith("?") or q.strip().endswith("？"):
                print("here")
                a_normalized = unicodedata.normalize("NFKC", a)
                if '。' in a_normalized:
                    split_index = a.rfind('。')
                    print(split_index)
                    if split_index == -1:
                        raise ValueError("No full stop found in answer")
                    parts = a.rsplit('。', 1)
                    if parts[0].strip():
                        flattened.extend([q, parts[0] + '。'])
                    if parts[1].strip():
                        flattened.extend([parts[1].strip()])
                else:
                    flattened.extend([q, a])
            else:
                flattened.extend([q, a])
        for i in flattened:
            if i.strip() and not i == "下載澳電APP 澳電微信":
                all_text += i + "\n"
        return
    
    for i in urls:
        tasks.append(asyncio.create_task(scrape_faq(i)))
    await asyncio.gather(*tasks)
    
    write_output(all_text, "cem_macau", "txt")
    return
    


if __name__ == "__main__":
    # asyncio.run(get_faq())
    # asyncio.run(get_questions_from_hk_elec())
    asyncio.run(get_us_faq())
    # asyncio.run(get_reddit_questions())
    # print(call_spider("https://zhidao.baidu.com/search?word=澳门%20电力"))
    # asyncio.run(crawl_zhidao_with_playwright())
    # print(call_spider("https://www.hkelectric.com/zh/customer-services/faqs/account-information"))