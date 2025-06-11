from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import AzureOpenAITextEmbedder, AzureOpenAIDocumentEmbedder
from haystack.components.generators import AzureOpenAIGenerator

class BasicRAG:
  def create_rag_pipeline():
    # Create main pipeline
    pipeline = Pipeline()
    document_store = InMemoryDocumentStore()
    # Add embedding and storage components
    pipeline.add_component("embedder", AzureOpenAIDocumentEmbedder(
        azure_endpoint= EMBED_ENDPOINT,
        azure_deployment= EMBED_DEPLOYMENT
    ))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))

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
        azure_endpoint=LLM_ENDPOINT,
        azure_deployment=LLM_DEPLOYMENT
    ))

    # Connect components for document storage
    pipeline.connect("embedder", "writer")
    # Connect components for querying
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")

    return pipeline

  def store_documents(pipeline, texts):
    documents = [Document(content=text) for text in texts]
    pipeline.run({"embedder": {"documents": documents}})

  def client_query(pipeline, query):
    response = pipeline.run({"prompt_builder": {"query": query}})
    return response["llm"]["replies"]