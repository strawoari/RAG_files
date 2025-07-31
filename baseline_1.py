from load_file import get_docs
import os, json, re, asyncio, numpy as np
from functools import partial
from pathlib import Path
import numpy as np
from openai import AsyncAzureOpenAI 
from haystack_pipeline import BasicRAG
import aiofiles

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

total_question_chars = 0
total_number_of_qa = 0
count_punc_answers = 0

def cosine_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

pattern = re.compile(
  r"question:(.*?)\nanswer:(.*?)\nref:(.*?)",
  re.DOTALL
)

punc_pattern = re.compile(r"[\u3000-\u303F\uFF00-\uFFEF!?.,]")

def get_qa_pairs(list_of_qa):
  matches = pattern.findall(list_of_qa)
  lst = []
  for question, answer, source in matches:
    lst.append({
      "question": question.strip(),
      "answer": answer.strip(),
      "source": source.strip()
    })
  return lst

embed_client = AsyncAzureOpenAI(
  api_key=AZURE_OPENAI_API_KEY,
  api_version=AZURE_EMBEDDING_API_VERSION,
  azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
)

async def run_in_thread(func, *args, **kwargs):
  loop = asyncio.get_running_loop()
  return await loop.run_in_executor(None, partial(func, *args, **kwargs))

async def async_client_query(rag, query: str) -> str:
  """Calls BasicRAG.client_query in a worker thread."""
  resp = await run_in_thread(rag.client_query, query)
  if isinstance(resp, (list, tuple)):
    return resp[0]
  else:
    print("Query response: " + resp)
    return resp

async def rate_answer(rag, qa_pair, sem: asyncio.Semaphore):
  question = qa_pair['question']
  total_question_chars += len(question)
  total_number_of_qa += 1
  answer = qa_pair['answer']
  count_punc_answers += len(punc_pattern.findall(answer))
  async with sem:
    response = await async_client_query(rag, question)
    # Move blocking embedding call to thread
    embedding = await run_in_thread(
        embed_client.embeddings.create,
        model=AZURE_EMBEDDING_DEPLOYMENT,
        input=[response, answer]
    )
  response_embedding, answer_embedding = [item.embedding for item in embedding.data]
  similarity = cosine_similarity(response_embedding, answer_embedding)
  return {
      "question": question,
      "response": response,
      "similarity": similarity,
  }

async def write_to_file(output_dir: Path, result, doc_url):
  idx = int(result['similarity'] * 10)
  file_path = output_dir / f"index_{idx}.txt"
  log = f"Query: {result['question']}\nResponse: {result['response']}\nDoc_url: {doc_url}\n\n"
  try:
      async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
          await f.write(log + "\n")
  except Exception as e:
      print(f"Error writing to {file_path}: {e}")
  return None

async def gather_all_answers(rag, input_dir: str, max_concurrency: int = 5):
  output_dir = Path("./outputs/baseline_1")
  output_dir.mkdir(parents=True, exist_ok=True)
  sem = asyncio.Semaphore(max_concurrency)
  tasks = []
  # Pre-create result files
  for i in range(10):
      file_path = output_dir / f"index_{i}.txt"
      file_path.touch(exist_ok=True)
  json_files = [Path(input_dir) / f for f in os.listdir(input_dir) if f.endswith('.json')]
  async def handle_one_qa(qa, rag, sem, doc_url, output_dir):
      try:
          result = await rate_answer(rag, qa, sem)
          await write_to_file(output_dir, result, doc_url)
      except Exception as e:
          print(f"Error processing QA pair: {e}")
  for fname in json_files:
      try:
          async with aiofiles.open(fname, "r") as f:
              raw = await f.read()
          json_data = json.loads(raw)
          for i, data in enumerate(json_data):
              list_of_qa = get_qa_pairs(data['list_of_qa'][0])
              for qa in list_of_qa:
                  tasks.append(asyncio.create_task(handle_one_qa(qa, rag, sem, data['doc_meta']['metadata']['source'], output_dir)))
      except Exception as e:
          print(f"Error reading file {fname}: {e}")
  await asyncio.gather(*tasks)
  return None

if __name__ == "__main__":
  rag = BasicRAG()
  index_pipeline = rag.create_index_pipeline()
  query_pipeline = rag.create_query_pipeline()
  # comment out if dataset rag already created document store!
  docs = get_docs('/Users/amychan/rag_files/data/web_data') 
  + get_docs('/Users/amychan/rag_files/data/pdf_docs')
  # comment out if dataset rag already created document store!
  asyncio.run(run_in_thread(rag.store_documents, index_pipeline, docs))
  print("Document store created, now running QA pairs")
  asyncio.run(
      gather_all_answers(rag, "/Users/amychan/rag_files/qa_data", max_concurrency=5)
  )
  print(f"Average question length: {float(total_question_chars) / float(total_number_of_qa)}")
  print(f"Average number of short phrases in answers: {float(count_punc_answers) / float(total_number_of_qa)}")
  # print(rag.client_query(query_pipeline, "澳门电力公司有哪些服务？"))
  