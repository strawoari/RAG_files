import asyncio
import json
import os
import shutil
from dotenv import load_dotenv
from pathlib import Path

from flexrag.datasets import RAGCorpusDataset, RAGCorpusDatasetConfig
from flexrag.models import EncoderConfig, OpenAIEncoderConfig
from flexrag.retriever import FlexRetriever, FlexRetrieverConfig
from flexrag.retriever.index import (
    FaissIndexConfig,
    MultiFieldIndexConfig,
    RetrieverIndexConfig,
)
from flexrag.retriever.index.bm25_index import BM25IndexConfig

load_dotenv()

# Manually clear the retriever path to ensure a fresh start
retriever_path = os.getenv("RETRIEVER_PATH")
if retriever_path and os.path.exists(retriever_path):
    print(f"Clearing existing retriever path: {retriever_path}")
    shutil.rmtree(retriever_path)

input_dir_1 = "/Users/amychan/rag_files/data/web_data"
input_dir_2 = "/Users/amychan/rag_files/data/pdf_docs"

dataset = RAGCorpusDataset(
    RAGCorpusDatasetConfig(
        file_paths= ["output_0.jsonl", "output_1.jsonl"],
        saving_fields=["source_file_path", "text"],
        encoding = "utf-8",
    )
)
print("Dataset done")
retriever = FlexRetriever(
    FlexRetrieverConfig(
        top_k = 5,
        log_interval=100000,
        batch_size=4096,
        retriever_path=os.getenv("RETRIEVER_PATH"),
        indexes_merge_method = "rrf"
    )
)
retriever.add_passages(passages=dataset)
print("Passages added")
retriever.add_index(
    index_name="bm25",
    index_config=RetrieverIndexConfig(
        index_type="bm25",
        bm25_config=BM25IndexConfig(
            lang="zh"
        ),
        passage_encoder_config=EncoderConfig(
            encoder_type="openai",
            openai_config=OpenAIEncoderConfig(
                is_azure=True,
                base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
                embedding_size=int(os.getenv("EMBEDDING_DIMENSION")),
# d keyword argument [type=unexpected_keyword_argument, input_value='1536', input_type=str]
            )
        )
    ),
    indexed_fields_config=MultiFieldIndexConfig(
        indexed_fields=["text"],
        merge_method="concat",
    )
)
print("BM25 index added")
retriever.add_index(
    index_name="contriever",
    index_config=RetrieverIndexConfig(
        index_type="faiss",  # specify the index type
        faiss_config=FaissIndexConfig(
            # let FaissIndex determine the index configuration automatically
            # you can also specify a specific index type like "Flat", "IVF", etc.
            index_type="FLAT",
            index_train_num=-1,  # use all available data for training
            n_probe=0,
            query_encoder_config=EncoderConfig(
                encoder_type="openai",  # specify using Hugging Face model
                openai_config=OpenAIEncoderConfig(
                    is_azure=True,
                    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
                    embedding_size=int(os.getenv("EMBEDDING_DIMENSION")),
                )
            ),
            passage_encoder_config=EncoderConfig(
                encoder_type="openai",  # specify using Hugging Face model
                openai_config=OpenAIEncoderConfig(
                    is_azure=True,
                    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
                    embedding_size=int(os.getenv("EMBEDDING_DIMENSION")),
                )
            ),
        ),
    ),
    indexed_fields_config=MultiFieldIndexConfig(
        indexed_fields=["text"],
        merge_method="concat",  # concatenate the `title` and `text` fields for indexing
    ),
)
print("FAISS dense index added")
retriever.save_to_local(os.getenv("RETRIEVER_PATH"))
print("Retriever saved")