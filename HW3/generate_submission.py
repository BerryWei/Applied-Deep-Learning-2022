import jsonlines
import os
import sys
       
with jsonlines.open(sys.argv[1], mode='w') as writer:
    with jsonlines.open(sys.argv[2]) as reader:
        with open('./tmp/tst-summarization/generated_predictions.txt', 'r') as f:
            titles = f.readlines()
            for row,title in zip(reader, titles):
                writer.write({'title': title, 'id': row['id']})
