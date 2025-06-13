import os
from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import AzureOpenAITextEmbedder, AzureOpenAIDocumentEmbedder
from haystack.components.generators import AzureOpenAIGenerator
from haystack.components.rankers import SentenceTransformersSimilarityRanker


AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

class BasicRAG(template = str):
  def __init__(self, template):
    self.template = template
    self.document_store = InMemoryDocumentStore()
    
  def create_index_pipeline(self):
    index_pipeline = Pipeline()
    index_pipeline.add_component(instance=DocumentCleaner(remove_extra_whitespaces = True), name="cleaner")
    index_pipeline.add_component(instance=DocumentSplitter(split_by="passage", split_length=5, language = "zh"), name="splitter")
    index_pipeline.add_component("embedder", AzureOpenAIDocumentEmbedder(
        azure_endpoint= AZURE_OPENAI_ENDPOINT,
        azure_deployment= AZURE_EMBEDDING_DEPLOYMENT,
    ))
    index_pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
    index_pipeline.connect("cleaner.documents", "splitter.documents")
    index_pipeline.connect("splitter.documents", "embedder.documents")
    index_pipeline.connect("embedder", "writer")
    return index_pipeline

  def create_query_pipeline(self):
    pipeline = Pipeline()
    query_embedder = AzureOpenAITextEmbedder(
      azure_endpoint=AZURE_OPENAI_ENDPOINT,
      azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    )
    ranker = SentenceTransformersSimilarityRanker()
    ranker.warm_up()
    pipeline.add_component("query_embedder", query_embedder)
    pipeline.add_component("retriever", ranker)
    pipeline.add_component(instance=ranker, name="ranker")
    pipeline.add_component("prompt_builder", PromptBuilder(template=self.template))
    pipeline.add_component("llm", AzureOpenAIGenerator(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT
    ))
    # Connect components for querying
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "ranker.documents")
    pipeline.connect("ranker.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    return pipeline

  def store_documents(self, texts):
    documents = [Document(content=text) for text in texts]
    self.index_pipeline.run({"embedder": {"documents": documents}})

  def client_query(self, query):
    response = self.query_pipeline.run(data = {
      "retriever": {"query": query, "top_k": 30},
      "ranker": {"query": query, "top_k": 5},
      "prompt_builder": {"query": query}
    })
    return response["llm"]["replies"]