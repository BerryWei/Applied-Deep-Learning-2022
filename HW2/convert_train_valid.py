import json
import os
import sys

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

with open('./data/context.json', 'r') as f:
    context_data = json.load(f)


for eachData in data:
    eachData['context'] = context_data[ eachData['relevant'] ]
    eachData['relevant'] = eachData['paragraphs'].index(eachData['relevant'])
    eachData['answer']['answer_start'] = [ eachData['answer']['start'] ]
    eachData['answer']['text'] = [ eachData['answer']['text'] ]
    eachData['answer'].pop('start')
    
    




os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
json.dump({'data': data}, open(sys.argv[2], 'w',encoding='utf-8'), indent=4, ensure_ascii=False)