# How to reproduce the **Chinese Question Answering** model
## 

```
┌── data
│   ├── context.json
│   ├── test.json
│   ├── test.json
│   └── valid.json
├── cache
│   ├── CC_prediction.json
│   ├── CC_processed_prediction.json
│   ├── test.json
│   └── valid.json
├── multiple-choice
├── question-answering
├── train_CC.sh
├── test_CC.sh
├── train_QA.sh
├── test_QA.sh
├── generate_csv.sh
├── convert.sh
├── convert_CC2QA.sh
└── *.py
```
## In this repo, we trained 2 models seperately and make a pipeline to do Chinese question answering .
    - Context Selection
    - Question Answering


##  1. Preprocess the data from  `./data` and the processed data would be saved in the `./cache` directory.
```bash=
bash convert.sh
```

##  2. Training Context Selection model
```bash=
bash train_CC.sh
```
##  3. Training Context Selection model
```bash=
bash train_QA.sh
```
## 4. Retrieve trained model and do the test pipeline thru 2 steps below.

1. You can download the trained model by executing `download.sh`
2. You can make prediction by executing `run.sh` pipline.
-  "${1}": path to the context file.
-  "${2}": path to the testing file.
-  "${3}": path to the output predictions.

```bash=
bash download.sh
bash run.sh "${1}"  "${2}"  "${3}"
```
