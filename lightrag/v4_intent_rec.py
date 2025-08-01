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
from sklearn.neighbors import NearestNeighbors
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

async def get_answer(query: str, context_response: str, sem, top_intent: str) -> tuple[str, str]:
    print(f"DEBUG: Getting answer for query: {query}")
    user_prompt = ""
    if not re.search(r'[\u4e00-\u9fff]', query):
        user_prompt = f"""---Role Description---

You are an intelligent assistant responsible for answering user questions about knowledge graphs and document fragments (provided in JSON format).

---Objective---

Generate a concise response based on Knowledge Base and follow Response Rules, considering the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When processing relationships that include timestamps, please follow these principles:
1. Each relationship contains a "created_at" timestamp, indicating when the knowledge was obtained.
2. When encountering conflicting relationships, consider both the semantic content and the timestamp.
3. Do not blindly prioritize the latest timestamp; make a judgment based on context.
4. For questions involving specific times, prioritize the time information in the content, then refer to the timestamp.

User Intent:
{top_intent}

---Knowledge Graph and Document Fragments---
{context_response}

---Answering Rules---
- Objective format and length: a response within a single paragraph, no more than 4 sentences.
- Use Markdown format and add appropriate headings.
- The response should be consistent with the "User Intent".
- The response language must match the user’s question language.
- In the "References" section, list at most 5 important sources, indicating whether they are from the knowledge graph (KG) or document fragments (DC), and note the file path in the following format:
[KG/DC] file_path
- If no answer is available, reply: "Currently, there is no relevant information. Please contact customer service or visit the official website."
- Do not fabricate information or introduce content not provided in the knowledge base.

User question:
{query}

Answer:
"""
    else:
        user_prompt = f"""---角色说明---

你是一名智能助手，负责回答用户关于知识图谱和文档片段（以 JSON 格式提供）的问题。

---目标---

根据知识库生成简洁的回复，并遵循下方“回答规则”。在作答时请参考用户提问。总结知识库中提供的所有相关信息，并可适当结合与知识库有关的通用知识。不得加入知识库中未提供的信息。

当处理包含时间戳的关系时，请遵循以下原则：
1.每条关系都包含一个 "created_at" 时间戳，表示我们获取该知识的时间；
2.在遇到冲突关系时，请同时考虑语义内容与时间戳；
3.不要盲目优先采用最新时间戳的关系，应根据上下文做出判断；
4.对于涉及具体时间的问题，优先考虑内容中的时间信息，其次才参考时间戳。

用户意图：
{top_intent}

---知识图谱与文档片段---
{context_response}

---回答规则---

- 目标格式与长度：一个段落内，不超过4句话
- 使用 Markdown 格式，并添加适当的标题
- 回复语言需与用户提问语言一致
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
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}])
        answer = completion.choices[0].message.content
        if not re.search(r'[\u4e00-\u9fff]', query):
            split_result = answer.split('References')
            if len(split_result) >= 2:
                main_content, refs = split_result[0], split_result[1]
            else:
                main_content, refs = answer, ""
            split_result = [main_content, refs]
        else:
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
            
            
async def get_embeddings(texts):
    if not texts:
        raise ValueError("get_embeddings called with empty input list")
    text_embeddings = await embed_client.embeddings.create(
        input=texts,
        model=AZURE_EMBEDDING_DEPLOYMENT,
    )
    # Check for empty or missing embeddings
    if not text_embeddings.data or not hasattr(text_embeddings.data[0], 'embedding') or len(text_embeddings.data[0].embedding) == 0:
        logging.error(f"No embeddings returned from API or embedding is empty for input: {texts}")
        return np.array([])
    return np.array([np.array(embedding.embedding) for embedding in text_embeddings.data])
texts = [
    ["为什么我的电费账单比平时高？", "你能解释一下账单上的收费项目吗？", "我该如何支付电费账单？", "有哪些支付方式可用？"], 
    ["为什么我的电力服务被中断？", "什么时候会恢复供电？", "我所在区域有计划停电吗？", "我如何报告停电情况？"], 
    ["我如何更改账户信息？", "我可以更新个人资料吗？", "我如何使用网上服务？", "我如何使用澳电软件？","如果我丢失了账单或合约号码该怎么办？", "我怎么投诉？"], 
    ["我如何减少电力使用？", "我可以获得能源消耗的明细吗？", "我如何读取电表？"], 
    ["我如何申请新的电力连接？", "我可以安装第二个电表吗？", "我如何升级服务容量？"], 
    ["我的电力闪烁，我该怎么办？", "我的电器因电力问题出现故障，可以获得支持吗？", "我能关闭断路器吗？"], 
    ["你们有哪些价格计划？", "你们是否为老年人或低收入家庭提供折扣？", "你们的高峰时段费率是多少？"], 
    ["你们是否提供绿色能源选项？", "我如何切换到可再生能源？", "太阳能源怎么安装？", "如何节约用电？"], 
    ["我如何报告倒下的电线？", "暴风雨期间使用电器安全吗？", "我如何检查家中是否存在电气隐患？"],
    ["我所在地区的用电规定是什么？", "客户需要遵守的供电政策有哪些？", "澳电如何处理欺诈行为？"]
]
labels = ["Billing Inquiry", "Service Interruption", "Account Management", "Energy Usage", "Service Application", "Technical Issues", "Electricity Rates and Plans", "Environmental Impact and Renewable Energy", "Safety Issues", "Regulations and Policy Consultation"]
labels_1 = []
texts_1 = []
for i in range(len(texts)):
        labels_1.extend([labels[i]] * len(texts[i]))
for i in range(len(texts)):
    texts_1.extend(texts[i])
example_embeddings = []
knn = NearestNeighbors(n_neighbors=2, metric='cosine')
from opencc import OpenCC
# Choose the conversion configuration: Traditional to Simplified
cc = OpenCC('t2s')
async def rag_fusion(query: str, sem):
    if re.search(r'[\u4e00-\u9fff]', query):
        query = cc.convert(query)
    query_embedding = await get_embeddings([query])
    if query_embedding.size == 0:
        logging.error(f"Empty embedding for query: {query}")
        return query, []
    query_embedding = query_embedding[0]
    # Use KNN for intent matching
    _, indices = knn.kneighbors(query_embedding.reshape(1, -1), n_neighbors=2)
    top_intent = " ".join([labels_1[index] for index in indices[0]])
    prompt = f"""You are helping clarify user queries for better document search.

User question: "{query}"

Intents: {top_intent}

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
            return sub_q_list, top_intent
        else:
            return [query], ""
    except Exception as e:
        logging.error(f"Error step back: {e}")
        return [query], ""

async def main() -> None:
    sem = asyncio.Semaphore(5)
    
    async def handle_one_qa(qa, index):
        try:
            step_back_queries, top_intents_string = await rag_fusion(qa['question'], sem)
            qa['step_back_query'] = " ".join(step_back_queries)
            qa['top_intent'] = top_intents_string
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
            split_result = await get_answer(qa['question'], string, sem, top_intents_string)
            if not split_result or not isinstance(split_result, (list, tuple)) or len(split_result) < 2 or not qa['answer']:
                logging.error(f"Skipping QA due to empty answer or split_result for question: {qa['question']}")
                return
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