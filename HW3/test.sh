TRAIN_FILE=./data/train.jsonl
VALID_FILE="${1}"


python run_summarization_no_trainer_pred.py \
    --model_name_or_path ./tmp/tst-summarization \
    --train_file ${TRAIN_FILE} \
    --validation_file ${VALID_FILE} \
    --summary_column title \
    --text_column maintext \
    --output_dir ./tmp/tst-summarization \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=16 \
    --learning_rate 1e-04 \
    --num_train_epochs 1 \
    --num_beams 5
	
