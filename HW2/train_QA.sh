TRAINFILE=./cache/train.json
VALIDFILE=./cache/valid.json
TESTFILE=./cache/CC_predction.json
OUTPUTDIR=./question-answering
CONTEXTFILE=./data/context.json



python run_qa_no_trainer.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --with_tracking \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 40 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir ${OUTPUTDIR} \
  --train_file ${TRAINFILE} \
  --validation_file ${VALIDFILE} \


