# How to reproduce the **Natural Language Generation** model
## 

```
┌── data
│   ├── public.jsonl
│   └── train.jsonl
├── run.sh
├── generate_submission.py
├── run_summarization_no_trainer_pred.py
├── run_summarization_no_trainer.py
├── test.sh
├── train.sh
└── eval.py
```
## In this repo, we use `google/mt5-small` pretrained model to automatically generate a title for a paragraph.


#  1. The training data is prepared in the `./data` folder.

#  2. Training NLG model
```bash=
bash train.sh
```
#  3. Retrieve trained model and do the testing pipeline thru 2 steps below.
1. You can download the trained model by executing `download.sh`
2. You can make prediction by executing `run.sh` pipline.
-  "${1}": path to the testing_input file
-  "${2}": path to the output predictions.
-  "${3}": path to the output predictions.
```bash=
bash download.sh
bash run.sh "${1}"  "${2}" 
```
#  4. evaluation
```bash=
usage: eval.py [-h] [-r REFERENCE] [-s SUBMISSION]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
  -s SUBMISSION, --submission SUBMISSION
``` 

