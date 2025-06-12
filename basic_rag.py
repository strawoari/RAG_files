from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import AzureOpenAITextEmbedder, AzureOpenAIDocumentEmbedder
from haystack.components.generators import AzureOpenAIGenerator
import os
from load_file import get_docs

AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

class BasicRAG:
  def __init__(self):
    self.query_pipeline = self.create_query_pipeline()
    self.index_pipeline = self.create_index_pipeline()
    self.document_store = InMemoryDocumentStore()
  
  def create_query_pipeline(self):
    # Create main pipeline
    pipeline = Pipeline()
    # Add embedding and storage components
    pipeline.add_component("embedder", AzureOpenAIDocumentEmbedder(
        azure_endpoint= AZURE_EMBEDDING_ENDPOINT,
        azure_deployment= AZURE_EMBEDDING_DEPLOYMENT
    ))
    pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
    # Connect components for document storage
    pipeline.connect("embedder", "writer")
    return pipeline

  def create_index_pipeline(self):
    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=self.document_store))
    template = """
你是一位电力工程顾问，专注于澳门电力的建设。
请用简短明了的语言回答以下客户问题：
背景信息:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

问题：{{ query }}
"""
    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component("llm", AzureOpenAIGenerator(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT
    ))
    # Connect components for querying
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    return pipeline

  def store_documents(self, pipeline, texts):
    documents = [Document(content=text) for text in texts]
    self.index_pipeline.run({"embedder": {"documents": documents}})

  def client_query(self, pipeline, query):
    response = self.query_pipeline.run({"prompt_builder": {"query": query}})
    return response["llm"]["replies"]

if __name__ == "__main__":
  rag = BasicRAG()
  docs = get_docs('/Users/amychan/rag_files/data')
  rag.store_documents(rag.index_pipeline, docs)
  print(rag.client_query(rag.query_pipeline, "test"))