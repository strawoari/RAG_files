from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.embedders import AzureOpenAITextEmbedder, AzureOpenAIDocumentEmbedder
from haystack.components.generators import AzureOpenAIGenerator
from bs4 import BeautifulSoup
import requests
import json
import os
from haystack.components.adapters import OutputAdapter
from langchain_community.document_loaders.sitemap import SitemapLoader
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

sitemap_loader = SitemapLoader(
    web_path="https://www.cem-macau.com/sitemap.xml",
    filter_urls=["https://www.cem-macau.com/zh(?!/press-release).*"]
)
docs = sitemap_loader.load()

# docs needs to be transformed into docs = List[Text]

### Set up index pipeline
document_store = InMemoryDocumentStore()
index_pipeline = Pipeline()
index_pipeline.add_component(instance=DocumentCleaner(), name="cleaner")
index_pipeline.add_component(instance=DocumentSplitter(split_by="sentence", split_length=1), name="splitter")
index_pipeline.add_component("embedder", AzureOpenAIDocumentEmbedder(
    azure_endpoint= AZURE_EMBEDDING_ENDPOINT,
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
))
index_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
index_pipeline.connect("cleaner.documents", "splitter.documents")
index_pipeline.connect("splitter.documents", "embedder.documents")
index_pipeline.connect("embedder", "writer")

### Set up query pipeline
pipeline = Pipeline()
template = """
你是一位问答助手。
根据以下背景信息与相关知识，请列出三个现实合理的客户问题，并给出相应的答案。
背景信息: 
{{ context }}
相关知识：
{% for document in documents %}
    {{ document.content }}
{% endfor %}
请确保：
1. 问题必须与文档中的具体信息、功能、流程、政策或概念直接相关。
2. 生成的问题包含多种类型。
4. 问题应清晰、简洁、口语化，模仿真实客户的提问方式。
6. 为每个生成的问题，简要说明它的出处，或关联了文档中的哪个具体知识点或章节。
请将你的回答以 JSON 格式返回，包含以下字段：
question（问题）和 answer（答案）和reference（出处）
"""
pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
pipeline.add_component("prompt_builder", PromptBuilder(template=template))
pipeline.add_component("llm", AzureOpenAIGenerator(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT
))
pipeline.add_component("json_parser", OutputAdapter(
    template="{{ llm.responses[0] | tojson | fromjson }}",
    output_type=dict,
))

# Connect components for querying
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")
pipeline.connect("llm", "json_parser")

### 7. Run Pipelines and Collect QA Pairs
qa_dataset = []

for doc in docs:
    document = Document(content=doc)
    # First run the indexing pipeline to process and store the document
    index_pipeline.run({"cleaner": {"documents": [document]}})
    # Then run the query pipeline to generate QA pairs
    result = pipeline.run({"prompt_builder":{"context": doc}})
    try:
        qa = result["json_parser"]["output"]  # Get the parsed JSON response
        qa_dataset.append(qa)
    except Exception as e:
        print("Failed to parse QA:", e)

### 8. Save to File
with open("qa_dataset.jsonl", "w", encoding="utf-8") as f:
    for qa in qa_dataset:
        json.dump(qa, f, ensure_ascii=False)
        f.write("\n")