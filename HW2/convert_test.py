import json
import os
import sys

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

with open('./data/context.json', 'r') as f:
    context_data = json.load(f)

for eachData in data:
    eachData['relevant'] = 0
    eachData['answer'] = {"text": ["XXX"], "answer_start" : [0]}
    # <<- eachData['context'] = context_data[ eachData['relevant'] ]
    eachData['context'] = "XXX"

    



os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
json.dump({'data': data}, open(sys.argv[2], 'w',encoding='utf-8'), indent=4, ensure_ascii=False)