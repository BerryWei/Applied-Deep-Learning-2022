# "${1}" is the first argument passed to the script --> 
# "${2}" is the second argument passed to the script -->
#testPath=/Users/berrywei/Documents/Python/ADL21-HW1/data/intent/test.json
ckptPath=./ckpt/intent/


# "${1}" /Users/berrywei/Documents/Python/ADL21-HW1/data/intent/test.json  
# "${2}" ./pred.intent.csv
python3 test_intent.py --test_file "${1}" --ckpt_path "${ckptPath}" --pred_file "${2}"