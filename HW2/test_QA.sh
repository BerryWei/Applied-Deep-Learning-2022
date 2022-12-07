TRAINFILE=./cache/train.json
VALIDFILE=./cache/valid.json
OUTPUTDIR=./question-answering
CONTEXTFILE="${1}"
TESTFILE=./cache/CC_processed_prediction.json
PREDFILE="${3}"


python run_qa_no_trainer.py \
  --model_name_or_path question-answering \
  --do_predict \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 0 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir ${OUTPUTDIR} \
  --train_file ${TRAINFILE} \
  --validation_file ${VALIDFILE} \
  --test_file ${TESTFILE} \


