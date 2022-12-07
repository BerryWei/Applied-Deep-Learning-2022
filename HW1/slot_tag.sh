# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
# testPath=/Users/berrywei/Documents/Python/ADL21-HW1/data/slot/test.json
ckptPath=./ckpt/slot/
# predPath=./pred.slot.csv

python3 test_slot.py --test_file "${1}" --ckpt_path "${ckptPath}" --pred_file "${2}"
