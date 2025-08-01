import asyncio
import json
import logging
from pathlib import Path
import aiofiles

from evaluation import spacy_score
from flexrag.retriever import FlexRetriever, FlexRetrieverConfig
from flexrag.retriever.index import (
    FaissIndexConfig,
    MultiFieldIndexConfig,
    RetrieverIndexConfig,
)
from flexrag.retriever.index.bm25_index import BM25IndexConfig

import numpy as np
from torch import cosine_similarity
from flexrag import prompt
from flexrag.assistant import ASSISTANTS, ModularAssistantConfig, AssistantBase
from flexrag.models import OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.context_refine import ContextArrangerConfig, ContextArranger
from flexrag.retriever.flex_retriever import FlexRetriever
from flexrag.text_process import ChineseSimplifier
from dotenv import load_dotenv
from flexrag.utils import configure
import os
from openai import AsyncAzureOpenAI
import sys
import os


load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("LIGHT_LLM_ENDPOINT")

AZURE_EMBEDDING_ENDPOINT = os.getenv("LIGHT_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("OPENAI_API_VERSION")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs/lightrag1")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

default_answer = "目前没有有关资料，请咨询客服人员或查询官网。"

generator = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)
embed_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_EMBEDDING_API_VERSION,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
)

myAssistantConfig = ModularAssistantConfig(
    generator_type = "openai",
    openai_config=OpenAIGeneratorConfig(
        is_azure=True,
        base_url= os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version= os.getenv("AZURE_OPENAI_API_VERSION"),
        model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
)

prompt_str = "你是一位澳门电力公司客户服务人员。\n请根据背景信息简短明了地回答客户问题，不超过4句。"

@ASSISTANTS("personal", config_class=ModularAssistantConfig)
class MyAssistant(AssistantBase):
    def __init__(self, config: ModularAssistantConfig):
        self.generator = OpenAIGenerator(config.openai_config)
        self.retriever = FlexRetriever.load_from_local(os.getenv("RETRIEVER_PATH"))
        self.query_rewriter = ChineseSimplifier()
        self.arranger = ContextArranger(ContextArrangerConfig(order = 'side'))
        self.prompt = ChatPrompt(system = ChatTurn(role="system", content=prompt_str))
        return

    def answer(self, question: str) -> str:
        # print(f"Searching for {question}")
        context = self.retriever.search(question)[0][:5]
        print(f"number of context: {len(context)}")
        # print(f"context: {context}")
        context = self.arranger.refine(context)
        # print(f"number of context after arranger: {len(context)}")  
        prompt_str = f"问题：{question}\n\n"
        for n, ctx in enumerate(context):
            prompt_str += f"背景信息 {n}: {ctx.data['text']}\n"
        # print(f"prompt_str: {prompt_str[:10]}")
        # Create a fresh prompt for each question
        current_prompt = ChatPrompt(system=ChatTurn(role="system", content=prompt_str))
        current_prompt.update(ChatTurn(role="user", content=prompt_str))
        # print(f"prompt: {current_prompt}")
        response = self.generator.chat([current_prompt])[0][0]
        # print(f"prompt after response: {current_prompt}")
        if len(context) >= 3:
            # print(f"context: {context[:3]}")
            return response, context[:3]
        else:
            # print(f"context: {context}")
            return response, context
    
myAssistant = MyAssistant(myAssistantConfig)

async def rate_answer(qa_pair, sem: asyncio.Semaphore):
    question = qa_pair['question']
    answer = qa_pair['answer']
    async with sem:
        response, contexts = await asyncio.to_thread(myAssistant.answer, question)
        print(f"Response: {response}")
        # Move blocking embedding call to thread
    contexts = [
        {"text": context.data['text'], 
         "source": context.data['source_file_path'], 
         "score": context.score}
        for context in contexts
    ]
    answer_score = spacy_score(response, answer)
    return {
        "question": question,
        "response": response,
        "answer": answer,
        "answer_score": answer_score,
        "contexts": contexts
    }
    
async def write_to_file(result, index):
    async with aiofiles.open(OUTPUT_DIR / f"flexrag_baseline.jsonl", "a") as f:
        await f.write(json.dumps(result, ensure_ascii=False) + "\n")

async def main() -> None:
    sem = asyncio.Semaphore(5)
    
    async def handle_one_qa(qa, index):
        try:
            result = await rate_answer(qa, sem)
            await write_to_file(result, index)
        except Exception as e:
            logging.error(f"Error handling QA {index}: {e}")
            # Write error to output for tracking
            await write_to_file(f"Error: {str(e)}", index)
            
    coros = []
    async with aiofiles.open("/Users/amychan/rag_files/lightrag/outputs/v0/annotated.json", "r") as f:
        raw = await f.read()
    json_data = json.loads(raw)
    for _, data in enumerate(json_data):
        q = data['Query']
        a = data['Answer']
        index = data['index']
        coros.append(handle_one_qa({'question': q, 'answer': a}, index))
    await asyncio.gather(*coros)