# How to train the model
##
##  Prepare the training data in the `./data/intent` and `./data/slot` as belows.

```
┌── data
│   ├── intent
│   │     ├── train.json
│   │     ├── eval.json
│   │     └── test.json
│   └── slot
│        ├── train.json
│        ├── eval.json
│        └── test.json
├── train_intent.py
└── train_slot.py
```
##  Modify model parameters and setting in `train_intent.py`, `train_slot.py` and  `model.py`


##  Training model
```bash=
python train_intent.py
python train_slot.py
```

##  Retrieve trained model and test 
```bash=
bash intent_cls.sh --test_file "${1}"  --pred_file "${2}"
bash slot_tag.sh --test_file "${1}"  --pred_file "${2}"
```