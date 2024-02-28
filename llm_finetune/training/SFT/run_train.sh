#!/bin/bash

INPUT_MODEL=$1
OUTPUT=$2
ZERO_STAGE=$3

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


INPUT_DATA="/data/li.sizhen/data/train_data_sep.v4.train.new_version.jsonl"
#INPUT_DATA="/data/li.sizhen/data/test_data.jsonl"
EVAL_DATA="/data/li.sizhen/data/train_data_sep.v4.eval.new_version.jsonl"

mkdir -p $OUTPUT

deepspeed main.py \
   --train_file "${INPUT_DATA}" \
   --eval_file "${EVAL_DATA}" \
   --model_name_or_path ${INPUT_MODEL} \
   --per_device_train_batch_size 20 \
   --per_device_eval_batch_size 20 \
   --max_seq_len 2048 \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --gradient_checkpointing \
   --only_optimize_lora \
   --lora_dim 0 \
   --lora_alpha 16 \
   --lora_learning_rate 5e-4 \
   --lora_module_name model.layers. \
   --deepspeed \
   --zero_stage $ZERO_STAGE \
   --output_dir $OUTPUT  \
   2>&1 | tee ${OUTPUT}/training.log

   #--offload \
