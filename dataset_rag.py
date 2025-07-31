from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.embedders import AzureOpenAITextEmbedder, AzureOpenAIDocumentEmbedder
from haystack.components.generators import AzureOpenAIGenerator
from haystack.utils import Secret
import json
import os
from load_file import get_docs
import concurrent.futures
from dotenv import load_dotenv
from functools import partial
import nltk
nltk.data.path.append('/Users/amychan/nltk_data')
nltk.download('punkt')
load_dotenv()


AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

### Set up index pipeline
def get_index_pipeline(document_store):
  index_pipeline = Pipeline()
  index_pipeline.add_component(instance=DocumentCleaner(remove_extra_whitespaces = True), name="cleaner")
  index_pipeline.add_component(instance=DocumentSplitter(split_by="page", split_length=5), name="splitter")
  index_pipeline.add_component("embedder", AzureOpenAIDocumentEmbedder(
    azure_endpoint=os.getenv("EMBEDDING_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    api_key=Secret.from_env_var("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
  ))
  index_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
  index_pipeline.connect("cleaner.documents", "splitter.documents")
  index_pipeline.connect("splitter.documents", "embedder.documents")
  index_pipeline.connect("embedder", "writer")
  return index_pipeline

### Set up query pipeline
def get_query_pipeline(document_store):
  pipeline = Pipeline()
  template = """
你是一位问答助手。
根据以下背景信息与相关知识，请列出{{ item_amount }}个现实合理的基于以下内容的客户问题，并给出相应的答案，答案应当简洁明了，不超过4句。
背景信息: 
{{ context }}
相关知识：
{% for document in documents %}
    {{ document.content }}
{% endfor %}
请确保：
1. 问题必须与文档中的具体信息、功能、流程、政策或概念直接相关。
2. 生成的问题包含多种类型。
3. 问题应清晰、简洁、口语化，模仿真实客户的提问方式。
4. 为每个生成的问题，简要说明它的出处，或关联了文档中的哪个具体知识点或章节，和文档的链接。
请按照以下格式返回：
question: ...
answer: ...
ref: ...
"""
  query_embedder = AzureOpenAITextEmbedder(
    azure_endpoint=os.getenv("EMBEDDING_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    api_key=Secret.from_env_var("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
  )
  pipeline.add_component("query_embedder", query_embedder)
  pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
  pipeline.add_component("prompt_builder", PromptBuilder(template=template))
  pipeline.add_component("llm", AzureOpenAIGenerator(
      azure_endpoint=AZURE_OPENAI_ENDPOINT,
      azure_deployment=AZURE_OPENAI_DEPLOYMENT
  ))
  # Connect components for querying
  pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
  pipeline.connect("retriever.documents", "prompt_builder.documents")
  pipeline.connect("prompt_builder", "llm")
  return pipeline

def process_doc(index_pipeline, query_pipeline, doc, file_name):
  print(type(doc.content))
  if not isinstance(doc.content, str):
    raise ValueError("Document content is not a string")
  # First run the query pipeline to generate QA pairs
  try:
    # First run the query pipeline to generate QA pairs
    result = query_pipeline.run({
        "query_embedder": {"text": doc.content},
        "prompt_builder": {"context": doc.content, "item_amount": (len(doc.content) / 400 +1) * 1}
    })
    qa = result["llm"]["replies"]  # Get the parsed JSON response
  except Exception as e:
    print("Failed to parse QA:", e)
    # print(doc.content)
    print(f"length of doc.content: {len(doc.content)}")
    raise e
  # Then run the indexing pipeline to process and store the document
  index_pipeline.run({"cleaner": {"documents": [doc]}})
  return {'list_of_qa': qa,  
          'doc_url': doc.meta['source'], 
          'doc_name': file_name}

docs, file_names = get_docs('/Users/amychan/rag_files/data/pdf_docs')

document_store = InMemoryDocumentStore()
index_pipeline = get_index_pipeline(document_store)
query_pipeline = get_query_pipeline(document_store)

print("Indexing pipeline and query pipeline are set up.")
### 7. Run Pipelines and Collect QA Pairs

process_fn = partial(process_doc, index_pipeline, query_pipeline)
max_workers = min(6, os.cpu_count() or 1)
print(f"Using {max_workers} workers for processing documents.")
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
  results = list(executor.map(process_fn, docs, file_names))
if results is None or len(results) == 0:
  print("No results returned from processing documents.")
else:
  print(results[0])

os.makedirs("qa_data", exist_ok=True)
for i in range(0, len(results), 40):
  with open(f"qa_data/qa_{i+160}-{i + 200}.json", "w", encoding='utf-8') as f:
    if i + 40 > len(results):
      json.dump(results[i:], f, ensure_ascii=False, indent=4)
    else:
      json.dump(results[i:i+40], f, ensure_ascii=False, indent=4)