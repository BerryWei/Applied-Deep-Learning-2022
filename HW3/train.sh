TRAIN_FILE=./data/train.jsonl
VALID_FILE=./data/public.jsonl


python run_summarization_no_trainer.py \
    --model_name_or_path google/mt5-small \
    --train_file ${TRAIN_FILE} \
    --validation_file ${VALID_FILE} \
    --summary_column title \
    --text_column maintext \
    --output_dir ./tmp/tst-summarization_10epochs \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=4 \
    --learning_rate 1e-04 \
    --num_train_epochs 20 \
    --with_tracking

