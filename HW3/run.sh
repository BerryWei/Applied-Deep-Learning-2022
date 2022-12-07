# ${1}: path to the input file
# ${2}: path to the output file

TEST_FILE="${1}"
SUB_FILE="${2}"


bash test.sh ${TEST_FILE}
python generate_submission.py ${SUB_FILE} ${TEST_FILE}
python eval.py -r public.jsonl -s ${SUB_FILE}