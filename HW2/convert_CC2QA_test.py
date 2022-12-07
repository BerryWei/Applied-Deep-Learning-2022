import json
import os
import sys

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

with open('./data/context.json', 'r') as f:
    context_data = json.load(f)


for eachData in data['data']:
    eachData['context'] = context_data[ eachData['relevant'] ]
    
    
    




os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
json.dump(data, open(sys.argv[2], 'w',encoding='utf-8'), indent=4, ensure_ascii=False)