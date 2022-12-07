
TRAINFILE=./cache/train.json
VALIDFILE=./cache/valid.json
TESTFILE=./cache/test.json
OUTPUTDIR=./multiple-choice
CONTEXTFILE=./data/context.json



python run_swag_no_trainer.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --max_length 512 \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 8 \
  --output_dir ${OUTPUTDIR} \
  --train_file ${TRAINFILE} \
  --validation_file ${VALIDFILE} \
  --context_file ${CONTEXTFILE} \