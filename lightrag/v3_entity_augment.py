import os
import asyncio
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import logging
from openai import AsyncAzureOpenAI
import aiofiles
import json
import re
import httpx
from evaluation import spacy_score

logging.basicConfig(level=logging.INFO)

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

retrieval_eighty_percent = 0
retrieval_sixty_percent = 0
retrieval_forty_percent = 0
retrieval_twenty_percent = 0
retrieval_zero_percent = 0
answer_eighty_five_percent = 0
answer_sixty_percent = 0
answer_forty_percent = 0
answer_twenty_percent = 0
answer_zero_percent = 0
async def rate_and_write_output(qa, split_result, context_response:str, index) -> None:
    global answer_eighty_five_percent, answer_sixty_percent, answer_forty_percent, answer_twenty_percent, answer_zero_percent
    try:
        question = qa['question']
        main_content, refs = split_result[0], split_result[1]
        answer = qa['answer']
        file_path = OUTPUT_DIR / f'entity_augment.json'
        best_answer = ""
        score1 = 0
        for ans in answer:
            score = spacy_score(main_content, ans)
            if score1 < score:
                score1 = score
                best_answer = ans
        if score1 >= 0.85:
            answer_eighty_five_percent += 1
        elif score1 >= 0.6:
            answer_sixty_percent += 1
        elif score1 >= 0.4:
            answer_forty_percent += 1
        elif score1 >= 0.2:
            answer_twenty_percent += 1
        else:
            answer_zero_percent += 1
        entities_json_str = re.search(r"-----Entities\(KG\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        relationships_json_str = re.search(r"-----Relationships\(KG\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        docs_json_str = re.search(r"-----Document Chunks\(DC\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        entities = json.loads(entities_json_str.group(1)) if entities_json_str else []
        relationships = json.loads(relationships_json_str.group(1)) if relationships_json_str else []
        docs = json.loads(docs_json_str.group(1)) if docs_json_str else []
        log = {
            "Query": question,
            "Answer": best_answer,
            "Spacy Score": score1,
            "Rewritten Query": qa['augmented_query'],
            "Response": main_content,
            "referenced context": refs,
            "Contexts": {"Entities": entities, "Relationships": relationships, "Documents": docs},
            "Index": index
        }
        # print(f"DEBUG: About to open file {file_path}")
        async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
            # print(f"DEBUG: File opened successfully")
            await f.write(json.dumps(log, ensure_ascii=False, indent=4) + "\n")
            # print(f"DEBUG: Data written to file")
            print(f"Wrote to {file_path}")
        # print(f"DEBUG: File context manager closed")
    except Exception as e:
        print(f"DEBUG: Exception occurred: {e}")
        logging.error(f"Error writing to {file_path}: {e}")
    return

generator = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)
async def get_answer(query: str, context_response: str) -> tuple[str, str]:
    print(f"DEBUG: Getting answer for query: {query}")
    user_prompt = f"""---角色说明---

你是一名智能助手，负责回答用户关于知识图谱和文档片段（以 JSON 格式提供）的问题。

---目标---

根据知识库生成简洁的回复，并遵循下方“回答规则”。在作答时请同时参考对话历史与当前用户提问。总结知识库中提供的所有相关信息，并可适当结合与知识库有关的通用知识。不得加入知识库中未提供的信息。

当处理包含时间戳的关系时，请遵循以下原则：
1.每条关系都包含一个 "created_at" 时间戳，表示我们获取该知识的时间；
2.在遇到冲突关系时，请同时考虑语义内容与时间戳；
3.不要盲目优先采用最新时间戳的关系，应根据上下文做出判断；
4.对于涉及具体时间的问题，优先考虑内容中的时间信息，其次才参考时间戳。

---对话历史---

---知识图谱与文档片段---
{context_response}

---回答规则---

- 目标格式与长度：一个段落内，不超过4句话
- 使用 Markdown 格式，并添加适当的标题
- 回复语言需与用户提问语言一致
- 回答内容需与对话历史保持连贯
- 在“参考资料”部分列出最多 5 个最重要的信息来源，标明来源于知识图谱（KG）或文档片段（DC），并注明文件路径，格式如下：
[KG/DC] file_path
- 如果找不到答案，请回复：“目前没有有关资料，请咨询客服人员或查询官网。”
- 不得编造信息，不得引入知识库未提供的内容

用户提问：
{query}

回答：
"""
    try:    
        completion = await generator.chat.completions.create(
            model="deepseek/deepseek-r1-0528-qwen3-8b:free",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}])
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error getting answer: {e}")
        return default_answer

async def process_query(expanded_query: str, query: str, sem: asyncio.Semaphore) -> tuple[str, str]:
    # print(f"DEBUG: Processing query: {expanded_query}")
    url = "http://0.0.0.0:1024/query"
    payload = {
        "query": expanded_query, 
        "only_need_context": True,
        "only_need_prompt": False,
        "mode": "hybrid", 
        "response_type": "Single Paragraph",
        "top_k": 15,
        "max_token_for_global_context": 200,
        "max_token_for_local_context": 200,
        # "conversation_history": conversation_history, 
        # "history_turns": len(conversation_history) // 2,
        "conversation_history": [],
        "history_turns": 0,
    }
    headers = {"Content-Type": "application/json"}
    
    max_retries = 10
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            async with sem:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                context_response = response.json()["response"]
                if "no context" in context_response:
                    return default_answer, []
                print(f"DEBUG: Context response: {context_response[:10]}")
                answer = await get_answer(query, context_response)
                split_result = answer.split('参考资料')
                if len(split_result) >= 2:
                    main_content, refs = split_result[0], split_result[1]
                else:
                    main_content, refs = answer, ""
                split_result = [main_content, refs]
                # add_to_conversation_history(query, answer)
                return split_result, context_response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, query '{query}': {e}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed, query '{query}': {e}")
                return f"Error in run_query: {str(e)}", []

embed_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_EMBEDDING_API_VERSION,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
)

import opencc
converter = opencc.OpenCC('t2s.json')
async def translate_query(query: str, sem: asyncio.Semaphore):
    prompt = query
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', query)
    english_chars = re.findall(r'[A-Za-z]', query)
    chinese_query = ""
    english_query = ""
    if len(chinese_chars) > len(english_chars):
        chinese_query = converter.convert(query)
        prompt = f"""You are a query translation assistant for a retrieval-based question answering system. Your task is to translate user queries written in Chinese into precise and natural English, in a way that aligns closely with entity names, relations, and terminology typically used in source documents (e.g., manuals, FAQs, policy pages).
Follow these guidelines:
1.Preserve the user's original intent and meaning.
2.Use formal or commonly used expressions that are likely to appear in the documents (e.g., translate “交电费” as “pay electricity bill” or “how to pay my electricity bill”).
3.Do not add new information or assumptions.
4.Clarify ambiguous references in the original question only if they can be clearly inferred from context (e.g., "这个" → "the electronic bill" if that’s clearly the subject).
5.If the original query is already clear, just improve fluency and precision without excessive rewriting.
6.Your priority is accurate translation, not expansion.

Original query (in Chinese):
{query}

Translated query (in English):
"""
        async with sem:
            completion = await generator.chat.completions.create(
                model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}])
            english_query = completion.choices[0].message.content
    elif len(english_chars) > len(chinese_chars):
        english_query = query
        prompt = f"""You are a query rewriting assistant for a retrieval-based question answering system. Your task is to rewrite user queries in a way that aligns closely with entity names, relations, and terminology typically used in source documents (e.g., manuals, FAQs, policy pages).
Follow these guidelines:
1.Preserve the user's original intent and meaning.
2.Use formal or commonly used expressions that are likely to appear in the documents (e.g., translate “pay electricity bill” as “交电费” or “how to pay my electricity bill”).
3.Do not add new information or assumptions.
4.Clarify ambiguous references in the original question only if they can be clearly inferred from context (e.g., "the electronic bill" → "这个" if that’s clearly the subject).
5.If the original query is already clear, just improve fluency and precision without excessive rewriting.
6.Your priority is accurate translation, not expansion.

Original query (in English):
{query}

Translated query (in Simplified Chinese):
"""
        async with sem:
            completion = await generator.chat.completions.create(
                model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}])
            chinese_query = completion.choices[0].message.content
    return chinese_query, english_query
        
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

with open("/Users/amychan/rag_test_light/LightRAG/graph_labels_embedding.json", "r") as f:
    graph_labels = json.load(f)
import spacy
chinese_nlp = spacy.load("zh_core_web_sm")
english_nlp = spacy.load("en_core_web_sm")
async def augment_query(chinese_query: str, english_query: str):
    lst = []
    doc =  chinese_nlp(chinese_query)
    set1 = set()
    for token in doc:
        for _, label in enumerate(graph_labels):
            similarity = cosine_similarity(token.vector, label['embedding'])
            if similarity >= 0.55:
                if label['label'] not in set1:
                    set1.add(label['label'])
                    lst.append((label['label'], similarity))
    doc = english_nlp(english_query)
    for token in doc:
        for _, label in enumerate(graph_labels):
            similarity = cosine_similarity(token.vector, label['embedding'])
            if similarity >= 0.55:
                if label['label'] not in set1:
                    set1.add(label['label'])
                    lst.append((label['label'], similarity))
    string = ""
    if len(lst) > 20:
        lst.sort(key=lambda x: x[1], reverse=True)
        string = " ".join([x[0] for x in lst[:20]])
    else:
        string = " ".join([x[0] for x in lst])
    return string

async def main() -> None:
    sem = asyncio.Semaphore(5)
    
    async def handle_one_qa(qa, index):
        try:
            chinese_query, english_query = await translate_query(qa['question'], sem)
            augmented_query = await augment_query(chinese_query, english_query)
            augmented_query = english_query + " " + augmented_query
            qa['augmented_query'] = augmented_query
            split_result, context_response = await process_query(augmented_query, qa['question'], sem)
            await rate_and_write_output(qa, split_result, context_response, index)
        except Exception as e:
            print(f"Error processing QA pair: {e}")
            
    coros = []
    async with aiofiles.open("/Users/amychan/rag_files/lightrag/outputs/v0/annotated.json", "r") as f:
        raw = await f.read()
    json_data = json.loads(raw)
    for _, data in enumerate(json_data):
        q = data['Query']
        a = data['Answer']
        index = data['index']
        coros.append(handle_one_qa({"question": q, "answer": a}, sem, index = index))
    await asyncio.gather(*coros)

if __name__ == "__main__":
    asyncio.run(main())