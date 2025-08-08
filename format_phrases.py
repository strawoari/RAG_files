import csv
import json

data = []
with open('car-window.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0] == '':
            if row[1] == '' and row[2] == '':
                continue
            else:
                if row[1] != '':
                    data[-1]['chinese'].append(row[1])
                if row[2] != '':
                    data[-1]['english'].append(row[2])
        else:
            dict1 = {
                'intent': row[0],
                'chinese': [],
                'english': []
            }
            if row[1] != '':
                dict1['chinese'].append(row[1])
            if row[2] != '':
                dict1['english'].append(row[2])
            data.append(dict1)

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)