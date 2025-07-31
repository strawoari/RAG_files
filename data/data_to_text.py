import os
import json
from pathlib import Path
from opencc import OpenCC

json_files = [f for f in Path('/Users/amychan/rag_files/data/pdf_docs').glob('*.json')]
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        split_lines = data['page_content'].split('\n')
        topic = split_lines[-1]
        data['page_content'] = '\n'.join(split_lines[:-1])
        source = data['metadata']['source']
        data['metadata'] = {
            'source': source,
            'topic': topic
        }
    with open(file, 'w', encoding='utf-8') as f2:
        json.dump(data, f2, indent=4, ensure_ascii=False)

# Convert Traditional to Simplified
# cc = OpenCC('t2s')
# json_files = [f for f in Path('/Users/amychan/rag_files/data/pdf_docs').glob('*.json')]
# json_files += [f for f in Path('/Users/amychan/rag_files/data/web_data').glob('*.json')]

# for file in json_files:
#     with open(file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         simplified = cc.convert(data['page_content'])
#         data['page_content'] = simplified
#     with open(file, 'w', encoding='utf-8') as f2:
#         json.dump(data, f2, indent=4, ensure_ascii=False)

