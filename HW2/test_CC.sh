TRAINFILE=./cache/train.json
VALIDFILE=./cache/valid.json
OUTPUTDIR=./multiple-choice

CONTEXTFILE="${1}"
TESTFILE="${2}"
PREDFILE="${3}"



python run_swag_no_trainer_test.py \
  --model_name_or_path multiple-choice \
  --max_length 512 \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 1 \
  --output_dir ${OUTPUTDIR} \
  --train_file ${TRAINFILE} \
  --validation_file ${TESTFILE} \
  --context_file ${CONTEXTFILE} \