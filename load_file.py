from haystack.components.converters import JSONConverter
from haystack.dataclasses import ByteStream
import os

def get_docs(dir1):
    # docs needs to be list of Document objects
    json_files = [os.path.join(dir1, f) for f in os.listdir(dir1) if f.endswith('.json')]
    docs = []
    for json_file in json_files:
        # Read the file content
        with open(json_file, "r", encoding="utf-8") as f:
            json_str = f.read()
        # Convert to ByteStream for Haystack
        source = ByteStream.from_string(json_str)
        # Set up the converter: adjust content_key and extra_meta_fields as needed
        converter = JSONConverter(content_key="page_content", extra_meta_fields={"metadata"})
        # Run the converter
        results = converter.run(sources=[source])
        # Collect the Document objects
        docs.extend(results["documents"])
    return docs