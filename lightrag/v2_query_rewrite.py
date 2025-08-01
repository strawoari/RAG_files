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


async def rate_and_write_output(qa, response: str, contexts, index: int) -> None:
    query = qa['question']
    answer = qa['answer']
    rewritten_query = qa['rewritten_query']
    score = spacy_score(response, answer)
    file_path = OUTPUT_DIR / f"baseline_{type}.json"
    log = {
        "query": query,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "Spacy Score": score,
        "response": response,
        "contexts": contexts,
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
请根据以下背景信息简短明了地回答以下客户问题，请把答案写得简短明了, 不要超过4句话。
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
        "top_k": 15,
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

def cosine_similarity(a, b):
    try:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}")
        return 0.0
embed_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_EMBEDDING_API_VERSION,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
)
async def get_ranked_contexts(query: str, context_response: str, sem: asyncio.Semaphore):
    # print(f"DEBUG: query: {query}\ncontext_response: {context_response}")
    try:
        entities_json_str = re.search(r"-----Entities\(KG\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        relationships_json_str = re.search(r"-----Relationships\(KG\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        docs_json_str = re.search(r"-----Document Chunks\(DC\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
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
        logging.error(f"Error in get_ranked_contexts: {e}")
        return []

async def rewrite_query(query: str, sem: asyncio.Semaphore):
    prompt = query
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', query)
    english_chars = re.findall(r'[A-Za-z]', query)
    if len(chinese_chars) > len(english_chars):
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
    elif len(english_chars) > len(chinese_chars):
        prompt = f"""You are a query rewriting assistant for a retrieval-based question answering system. Your task is to rewrite user queries in a way that aligns closely with entity names, relations, and terminology typically used in source documents (e.g., manuals, FAQs, policy pages).
Follow these guidelines:
1.Preserve the user's original intent and meaning.
2.Use formal or commonly used expressions that are likely to appear in the documents (e.g., translate "pay electricity bill" as "how to pay my electricity bill").
3.Do not add new information or assumptions.
4.Clarify ambiguous references in the original question only if they can be clearly inferred from context (e.g., "the electronic bill" → "这个" if that’s clearly the subject).
5.If the original query is already clear, just improve fluency and precision without excessive rewriting.

Original query:
{query}

Rewritten query:
"""
    try:
        async with sem:
            completion = await generator.chat.completions.create(
                model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}])
            return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error expanding query: {e}")
        return query
        
async def main() -> None:
    sem = asyncio.Semaphore(5)
    
    async def handle_one_qa(qa, sem, index):
        try:
            rewritten_query = await rewrite_query(qa['question'], sem)
            response, contexts = await process_query(rewritten_query, rewritten_query, sem)
            qa['rewritten_query'] = rewritten_query
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