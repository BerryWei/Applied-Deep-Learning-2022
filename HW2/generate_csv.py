import json
import sys
import csv

with open(sys.argv[1], 'r') as f:
    data = json.load(f)


with open(sys.argv[2], 'w') as csvfile:  
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'answer'])
    for k, v in data.items():
       writer.writerow([k, v])