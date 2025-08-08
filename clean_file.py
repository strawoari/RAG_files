import json


with open('/Users/amychan/rag_files/data_expanded_all0.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    phrases = item['phrases']
    lst = []
    for phrase in phrases:
        phrase = phrase.strip()
        if '场景' in phrase or (len(phrase) <= 3 and '语句' in phrase):
            continue
        if phrase.startswith('语句'):
            phrase
        lst.append(phrase)
    item['phrases'] = lst
    

with open('/Users/amychan/rag_files/data_expanded_all0_clean.json', 'w', encoding='utf-8') as f:
    json.dump(list(set1), f, ensure_ascii=False, indent=4)