from haystack.components.converters import JSONConverter, MarkdownToDocument
from haystack.dataclasses import ByteStream
import os

def get_docs(dir1):
    # docs needs to be list of Document objects
    json_files = [os.path.join(dir1, f) for f in os.listdir(dir1) if f.endswith('.json')]
    all_docs = []
    # markdown_converter = MarkdownToDocument()
    file_names = []
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
        json_docs = results["documents"]
        print(json_docs[0])
        for doc in json_docs:
            doc.meta = doc.meta['metadata']
            for k, v in doc.meta.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    continue
                elif isinstance(v, list):
                    string = ''
                    for i in v:
                        string += i
                    doc.meta[k] = string
                else:
                    doc.meta[k] = str(v)
            file_names.append(json_file)
        #     md_content = doc.content  # this is a markdown string
        #     md_stream = ByteStream.from_string(md_content)
        #     md_result = markdown_converter.run(sources=[md_stream])
        #     for i in md_result['documents']:
        #         i.meta = doc.meta
        all_docs.extend(json_docs)
    return all_docs, file_names