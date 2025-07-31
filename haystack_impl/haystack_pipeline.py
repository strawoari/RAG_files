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
from haystack.components.rankers import LostInTheMiddleRanker
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder

AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

system_message = ChatMessage.from_system(
  "你是一位电力工程顾问，专注于澳门电力的建设。\n请用简短明了的语言回答以下客户问题。")

prompt_template = """
背景信息:
    {% for document in documents %}
    {{ document.content }}
    {% endfor %}
    对话：
    {% for conversation in conversation_history %}
    {{ conversation.question }}
    回答: {{ conversation.answer }}
    {% endfor %}
"""

class BasicRAG():
  def __init__(self):
    self.document_store = InMemoryDocumentStore()
    self.conversation_history = [system_message]
    
  def create_index_pipeline(self):
    index_pipeline = Pipeline()
    index_pipeline.add_component(instance=DocumentCleaner(remove_extra_whitespaces = True), name="cleaner")
    index_pipeline.add_component(instance=DocumentSplitter(split_by="page", split_length=1), name="splitter")
    index_pipeline.add_component("embedder", AzureOpenAIDocumentEmbedder(
      azure_endpoint=os.getenv("EMBEDDING_ENDPOINT"),
      azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
      api_key= Secret.from_env_var("AZURE_OPENAI_API_KEY"),
      api_version=os.getenv("OPENAI_API_VERSION"),
    ))
    index_pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))
    index_pipeline.connect("cleaner.documents", "splitter.documents")
    index_pipeline.connect("splitter.documents", "embedder.documents")
    index_pipeline.connect("embedder", "writer")
    return index_pipeline
  
  def add_conversation(self, content_type, content):
    if len(self.conversation_history) >= 5:
      self.conversation_history.pop(1)
    if content_type == "user":
      self.conversation_history.append(ChatMessage.from_user(content))
    else:
      self.conversation_history.append(ChatMessage.from_assistant(content))

  def create_query_pipeline(self):
    pipeline = Pipeline()
    query_embedder = AzureOpenAITextEmbedder(
      azure_endpoint=os.getenv("EMBEDDING_ENDPOINT"),
      azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
      api_key=Secret.from_env_var("AZURE_OPENAI_API_KEY"),
      api_version=os.getenv("OPENAI_API_VERSION"),
    )
    # ranker = SentenceTransformersSimilarityRanker()
    # ranker.warm_up()
    pipeline.add_component("query_embedder", query_embedder)
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(ldocument_store=self.document_store))
    pipeline.add_component("ranker", LostInTheMiddleRanker(),  outputs={"documents": "ranked_documents"})
    pipeline.add_component("prompt_builder", ChatPromptBuilder(
      template=prompt_template, required_variables={"documents", "conversation_history"}))
    pipeline.add_component("llm", AzureOpenAIGenerator(
      azure_endpoint=os.getenv("LLM_ENDPOINT"),
      azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
      api_key=Secret.from_env_var("AZURE_OPENAI_API_KEY"),
      api_version=os.getenv("OPENAI_API_VERSION"),
    ))
    # Connect components for querying
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "ranker.documents")
    pipeline.connect("ranker.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    return pipeline

  def store_documents(self, index_pipeline, texts):
    documents = [Document(content=text) for text in texts]
    index_pipeline.run({"embedder": {"documents": documents}})

  def client_query(self, query_pipeline, query):
    self.add_conversation("user", query)  
    response = query_pipeline.run(data = {
      "retriever": {"query": query, "top_k": 5},
      "prompt_builder": {"conversation_history": self.conversation_history}
    })
    print(response)
    response_message = response["llm"]["replies"]
    print(response_message)
    self.add_conversation("assistant", response_message)
    return response_message