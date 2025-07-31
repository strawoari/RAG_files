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

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

type = "hybrid"

embed_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_EMBEDDING_API_VERSION,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
)

async def rate_and_write_output(query: str, expanded_query: str, response: str, answer: str, index: int) -> None:
    split_result = answer.split('参考资料')
    main_content = response
    refs = ""
    if len(split_result) >= 2:
        main_content, refs = split_result[0], split_result[1]
    score = spacy_score(main_content, answer)
    file_path = OUTPUT_DIR / f"baseline_{type}.json"
    log = {
        "query": query,
        "expanded_query": expanded_query,
        "answer": answer,
        "response": main_content,
        "score": score,
        "index": index
    }
    try:
        async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(log, ensure_ascii=False, indent=4) + "\n")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
    return

generator = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)
async def expand_query(query: str, sem: asyncio.Semaphore):
    try:
        prompt = f"""你是一个查询重写助手，目标是在基于检索的问答系统中重写用户问题，使其尽可能匹配文档中的实体和关系，从而提升召回准确性。

请遵循以下原则进行改写：
- 明确模糊或省略的实体（如"这个"应改为"电子账单"或具体服务名）
- 补全用户未说出的但在上下文中暗示的要素（如地点"澳门"、时间"本月"、服务提供者"澳电"等）
- 将用户的问题表达方式改为文档中常见或正式的表达（例如"交电费" → "如何通过银行缴交电费"）
- 保留原始意图，**不要编造虚假信息或虚构实体**
- 如果原问题已清晰明确，可适当润色但不需大幅改写

原始问题：
{query}

重写后的问题：
"""
        async with sem:
            completion = await generator.chat.completions.create(
                model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}])
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error expanding query: {e}")
        return query  # Return original query if expansion fails

async def get_ranked_contexts(query: str, context_response: str, sem: asyncio.Semaphore):
    # print(f"DEBUG: query: {query}\ncontext_response: {context_response}")
    try:
        entities_json_str = re.search(r"-----Entities\(KG\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        relationships_json_str = re.search(r"-----Relationships\(KG\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        docs_json_str = re.search(r"-----Document Chunks\(DC\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        # print(f"DEBUG: entities_json_str: {entities_json_str}")
        # print(f"DEBUG: relationships_json_str: {relationships_json_str}")
        # print(f"DEBUG: docs_json_str: {docs_json_str}")
        contexts = []
        # Parse entities with error handling
        try:
            entities = json.loads(entities_json_str.group(1)) if entities_json_str else []
            print(f"DEBUG: Number of entities parsed: {len(entities)}")
            if entities and len(entities) > 0:
                print(f"DEBUG: entities: {entities[0]}")
            for entity in entities:
                try:
                    contexts.append({
                        "type": "entity",
                        "entity": entity.get('entity', ''),
                        "description": entity.get('description', ''),
                        "original_raw": entity.get('rank', 0),
                        "doc_loc": entity.get('file_path', '')
                    })
                    print(f"DEBUG: Added entity context, total contexts now: {len(contexts)}")
                except Exception as e:
                    logging.warning(f"Error processing entity: {e}")
                    continue
        except (json.JSONDecodeError, AttributeError) as e:
            logging.warning(f"Error parsing entities JSON: {e}")
        
        # Parse relationships with error handling
        try:
            relationships = json.loads(relationships_json_str.group(1)) if relationships_json_str else []
            print(f"DEBUG: Number of relationships parsed: {len(relationships)}")
            for relationship in relationships:
                try:
                    contexts.append({
                        "type": "relationship",
                        "entity1": relationship.get('entity1', ''),
                        "entity2": relationship.get('entity2', ''),
                        "description": relationship.get('description', ''),
                        "original_rank": relationship.get('rank', 0),
                        "source": relationship.get('file_path', '')
                    })
                    # print(f"DEBUG: Added relationship context, total contexts now: {len(contexts)}")
                except Exception as e:
                    logging.warning(f"Error processing relationship: {e}")
                    continue
        except (json.JSONDecodeError, AttributeError) as e:
            logging.warning(f"Error parsing relationships JSON: {e}")
        
        # Parse docs with error handling
        try:
            docs = json.loads(docs_json_str.group(1)) if docs_json_str else []
            print(f"DEBUG: Number of docs parsed: {len(docs)}")
            if docs and len(docs) > 0:
                print(f"DEBUG: docs: {docs[0]}")
            for doc in docs:
                try:
                    # Handle case where doc might be a string or dictionary
                    if isinstance(doc, str):
                        contexts.append({
                            "type": "doc",
                            "description": doc,
                            "source": "",
                            "doc_loc": ""
                        })
                        print("Context doc is a string and not json-formatted" + str(doc))
                    elif isinstance(doc, dict):
                        contexts.append({
                            "type": "doc",
                            "description": doc.get('content', {}).get('page_content', '') if isinstance(doc.get('content'), dict) else str(doc.get('content', '')),
                            "source": doc.get('content', {}).get('metadata', {}).get('source', '') if isinstance(doc.get('content'), dict) else "",
                            "doc_loc": doc.get('file_path', '')
                        })
                    else:
                        logging.warning(f"Unexpected doc type: {type(doc)}")
                        continue
                except Exception as e:
                    logging.warning(f"Error processing doc: {e}")
                    continue
        except (json.JSONDecodeError, AttributeError) as e:
            logging.warning(f"Error parsing docs JSON: {e}")
        
        # print(f"DEBUG: Total contexts before formatting: {len(contexts)}")
        context_formatted = []
        for i, context in enumerate(contexts):
            try:
                print(f"DEBUG: Formatting context {i}: {context['type']}")
                if context['type'] == 'entity':
                    context_formatted.append(f"{context['entity']}\n{context['description']}")
                elif context['type'] == 'relationship':
                    context_formatted.append(f"{context['entity1']}与{context['entity2']}\n{context['description']}")
                elif context['type'] == 'doc':
                    context_formatted.append(f"{context['description']}")
                # print(f"DEBUG: Added formatted context {i}, total formatted now: {len(context_formatted)}")
            except KeyError as e:
                logging.warning(f"Error formatting context {i}: {e}")
                continue
        print(f"DEBUG: Number of formatted contexts: {len(context_formatted)}")
        return context_formatted
    except Exception as e:
        logging.error(f"Error getting ranked contexts: {e}")
        return []

default_answer = "目前没有有关资料，请咨询客服人员或查询官网。"
async def get_answer(query: str, contexts: dict, sem: asyncio.Semaphore) -> tuple[str, str]:
    print(f"DEBUG: Getting answer for query: {query}")
    try:
        context_string = ""
        for i in range(len(contexts)):
            try:
                if contexts[i]['type'] == 'entity':
                    context_string += f"{i}. {contexts[i]['entity']}：{contexts[i]['description']}\n"
                elif contexts[i]['type'] == 'relationship':  # Fixed: was context[i]
                    context_string += f"{i}. {contexts[i]['entity1']}与{contexts[i]['entity2']}：{contexts[i]['description']}\n"
                elif contexts[i]['type'] == 'doc':  # Fixed: was context[i]
                    context_string += f"{i}. {contexts[i]['content']}\n"
            except (KeyError, IndexError) as e:
                logging.warning(f"Error processing context {i}: {e}")
                continue
        
        user_prompt = f"""
你是一位电力工程顾问，专注于澳门电力的建设。
请根据以下背景信息简短明了地回答以下客户问题，请只用大概两句话。
如果背景信息中没有相关信息，请回答'{default_answer}'
背景信息：
{context_string}

问题：{query}
"""
        async with sem:
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
    url = "http://localhost:1024/query"
    payload = {
        "query": expanded_query, 
        "only_need_context": True,
        "only_need_prompt": False,
        "mode": "hybrid", 
        "top_k": 7,
        "max_token_for_text_unit": 200,
        "max_token_for_global_context": 200,
        "max_token_for_local_context": 200,
        "response_type": "Single Paragraph"
    }
    headers = {"Content-Type": "application/json"}
    
    max_retries = 3
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            async with sem:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                res = response.json()["response"]
                print(f"result saved ({len(res)} chars)")
                if res.startswith("Error"):
                    return "Error in lightrag context retrieval API: " + res, []
                contexts = await get_ranked_contexts(query, res, sem)
                if len(contexts) == 0:
                    return default_answer, []
                print(f"DEBUG: query {query} got contexts {len(contexts)}")
                answer = await get_answer(query, contexts, sem)
                return answer, contexts
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, query '{query}': {e}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed, query '{query}': {e}")
                return f"Error in run_query: {str(e)}", []

async def main() -> None:
    sem = asyncio.Semaphore(5)
    
    async def handle_one_qa(qa, sem, index):
        try:
            response, contexts = await process_query(qa['question'], qa['question'], sem)
            await rate_and_write_output(qa, response, contexts, index)
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