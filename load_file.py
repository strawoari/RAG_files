import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile

def load_file_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        suffix = ".pdf" if url.lower().endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(response.content)
            f.flush()
            if suffix == ".pdf":
                loader = PyPDFLoader(f.name)
            else:
                loader = TextLoader(f.name, autodetect_encoding=True)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
            return docs
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return []