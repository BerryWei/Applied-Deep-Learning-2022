# "${1}": path to the context file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.
# Berry test: bash run.sh ./data/context.json ./test.json ./submission.csv

bash convert.sh "${2}"
bash test_CC.sh "${1}" ./cache/test.json  "${3}"
bash convert_CC2QA.sh
bash test_QA.sh "${1}" ./cache/test.json  "${3}"
bash generate_csv.sh "${3}"
