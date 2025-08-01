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

def cosine_similarity(a, b):
    try:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}")
        return 0.0
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
async def rate_and_write_output(qa, split_result, context_response:dict, index) -> None:
    global answer_eighty_five_percent, answer_sixty_percent, answer_forty_percent, answer_twenty_percent, answer_zero_percent
    try:
        question = qa['question']
        main_content, refs = split_result[0], split_result[1]
        answer = qa['answer']
        file_path = OUTPUT_DIR / f'stepback.json'
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
        log = {
            "Query": question,
            "Answer": best_answer,
            "Spacy Score": score1,
            "Step Back Query": qa['step_back_query'],
            "Response": main_content,
            "referenced context": refs,
            "Contexts": context_response,
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

async def get_answer(query: str, context_response: str, sem) -> tuple[str, str]:
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
        async with sem:
            completion = await generator.chat.completions.create(
                model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}])
        answer = completion.choices[0].message.content
        split_result = answer.split('参考资料')
        if len(split_result) >= 2:
            main_content, refs = split_result[0], split_result[1]
        else:
            main_content, refs = answer, ""
        split_result = [main_content, refs]
        return split_result
    except Exception as e:
        logging.error(f"Error getting answer: {e}")
        return default_answer, ""
    
def cosine_similarity_1(a, b):
    try:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}")
        return 0.0
async def fuse_and_rerank(context_list: list[str], query, sem):
    entity_relation_input = []
    docs = []
    entity_set = set()
    relationship_set = set()
    doc_set = set()
    for context_response in context_list:
        entities_json_str = re.search(r"-----Entities\(KG\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        relationships_json_str = re.search(r"-----Relationships\(KG\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        docs_json_str = re.search(r"-----Document Chunks\(DC\)-----\s*```json\n(.*?)\n```", context_response, re.DOTALL)
        entities_list = json.loads(entities_json_str.group(1)) if entities_json_str else []
        relationships_list = json.loads(relationships_json_str.group(1)) if relationships_json_str else []
        docs_list = json.loads(docs_json_str.group(1)) if docs_json_str else []
        for entity in entities_list:
            if entity['entity'] not in entity_set:
                entity_set.add(entity['entity'])
                entity_relation_input.append(entity)
        for relationship in relationships_list:
            if (relationship['entity1'], relationship['entity2']) not in relationship_set:
                relationship_set.add((relationship['entity1'], relationship['entity2']))
                entity_relation_input.append(relationship)
        for doc in docs_list:
            if doc['file_path'] not in doc_set:
                doc_set.add(doc['file_path'])
                docs.append(doc)
    async with sem:
        query_embedding = await embed_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        query_embedding = query_embedding.data[0].embedding
        print(f"DEBUG: query_embedding length: {len(query_embedding)}")
        entity_relation_embeddings = await embed_client.embeddings.create(
            input=[item['description'] for item in entity_relation_input],
            model="text-embedding-3-small"
        )
        entity_relation_embeddings = [data.embedding for data in entity_relation_embeddings.data]
        print(f"DEBUG: entity_relation_embeddings count: {len(entity_relation_embeddings)}")
        doc_embeddings = await embed_client.embeddings.create(
            input=[doc['content'] for doc in docs],
            model="text-embedding-3-small"
        )
        doc_embeddings = [data.embedding for data in doc_embeddings.data]
        print(f"DEBUG: doc_embeddings count: {len(doc_embeddings)}")
    top_entities = []
    top_relations = []
    top_docs = []
    entity_relation_score = []
    try:
        for i in range(len(entity_relation_embeddings)):
            score = cosine_similarity_1(query_embedding, entity_relation_embeddings[i])
            entity_relation_score.append([score, i])
        print(f"DEBUG: entity_relation_score (first 5): {entity_relation_score[:5]}")
        entity_relation_score.sort(key=lambda x: x[0], reverse=True)
        print(f"DEBUG: entity_relation_score sorted (first 5): {entity_relation_score[:5]}")
        for i in range(min(15, len(entity_relation_score))):
            index = entity_relation_score[i][1]
            # print(f"DEBUG: entity_relation_input[{index}]: {entity_relation_input[index]}")
            if 'entity' in entity_relation_input[index]:
                top_entities.append(entity_relation_input[index])
            else:
                top_relations.append(entity_relation_input[index])
    except Exception as e:
        print(f"DEBUG: Exception in entity/relationship scoring: {e}")
        raise
    doc_score = []
    try:
        for i in range(len(doc_embeddings)):
            score = cosine_similarity_1(query_embedding, doc_embeddings[i])
            doc_score.append([score, i])
        # print(f"DEBUG: doc_score (first 5): {doc_score[:5]}")
        doc_score.sort(key=lambda x: x[0], reverse=True)
        print(f"DEBUG: doc_score sorted (first 5): {doc_score[:5]}")
        for i in range(min(10, len(doc_score))):
            index = doc_score[i][1]
            # print(f"DEBUG: docs[{index}]: {docs[index]}")
            top_docs.append(docs[index])
        top_docs.reverse()
    except Exception as e:
        print(f"DEBUG: Exception in doc scoring: {e}")
        raise
    return top_entities, top_relations, top_docs

async def process_query(query: str, sem: asyncio.Semaphore) -> tuple[str, str]:
    # print(f"DEBUG: Processing query: {expanded_query}")
    url = "http://0.0.0.0:1024/query"
    payload = {
        "query": query, 
        "only_need_context": True,
        "only_need_prompt": False,
        "mode": "hybrid", 
        "response_type": "Single Paragraph",
        "top_k": 15,
        "max_token_for_global_context": 200,
        "max_token_for_local_context": 200,
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
                # context_response = add_to_context_response(context_response)
                print(f"DEBUG: Context response: {context_response[:10]}")
                return context_response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, query '{query}': {e}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed, query '{query}': {e}")
                return f"Error in run_query: {str(e)}", []

async def rag_fusion(query: str, sem):
    prompt = f"""You are helping clarify user queries for better document search.

User question: "{query}"

Step back: What is this question really asking? What information would help answer it accurately? Please follow the following rules:
1. If the original question is already specific and directly answerable by documents or the knowledge graph, do not split it. Just repeat it as the only sub-question.
2. Output the sub-questions in English.
3. The sub-questions should be more specific than the 'User question'.
4. The sub-questions should be more likely to be answered by the documents.
5. The sub-questions should be more likely to be answered by the knowledge graph.
6. If you are unsure, prefer fewer but higher-quality sub-questions.

Output strictly in the following format and nothing else:
Sub-questions:
Question 1.
Question 2.
End of sub-questions.

Do not explain your reasoning. Only output the sub-questions as specified.
"""
    try:
        async with sem:
            completion = await generator.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}])
        response = completion.choices[0].message.content
        if "Sub-questions:" in response:
            sub_q_string = response.split("Sub-questions:")[1].split("End of sub-questions.")[0].strip()
            sub_q_list = sub_q_string.split("\n")
            sub_q_list = [q.strip() for q in sub_q_list if q.strip()]
            return sub_q_list
        else:
            return query
    except Exception as e:
        logging.error(f"Error step back: {e}")
        return query

async def main() -> None:
    sem = asyncio.Semaphore(5)
    
    async def handle_one_qa(qa, index):
        try:
            step_back_queries = await rag_fusion(qa['question'], sem)
            qa['step_back_query'] = " ".join(step_back_queries)
            context_list = []
            for i in range(len(step_back_queries)):
                context_response = await process_query(step_back_queries[i], sem)
                context_list.append(context_response)
            entities, relationships, docs = await fuse_and_rerank(context_list, qa['question'], sem)
            string = "-----Entities(KG)-----"
            string += "\n" + json.dumps(entities, ensure_ascii=False, indent=4)
            string += "\n" + "-----Relationships(KG)-----"
            string += "\n" + json.dumps(relationships, ensure_ascii=False, indent=4)
            string += "\n" + "-----Document Chunks(DC)-----"
            string += "\n" + json.dumps(docs, ensure_ascii=False, indent=4)
            string += "\n\n"
            print(f"DEBUG: String: {string[:20]}")
            split_result = await get_answer(qa['question'], string, sem)
            await rate_and_write_output(qa, split_result, 
                                        {'entities': entities, 'relationships': relationships, 'docs': docs}, index)
        except Exception as e:
            logging.error(f"Error handling QA {index}: {e}")
            # Write error to output for tracking
            await rate_and_write_output(qa, f"Error: {str(e)}", [], index)
            
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