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


INPUT_DATA=""
EVAL_DATA=""

mkdir -p $OUTPUT

deepspeed main.py \
   --train_file "${INPUT_DATA}" \
   --eval_file "${EVAL_DATA}" \
   --model_name_or_path "${INPUT_MODEL}" \
   --per_device_train_batch_size 6 \
   --per_device_eval_batch_size 6 \
   --max_seq_len 2048 \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --only_optimize_lora \
   --zero_stage $ZERO_STAGE \
   --lora_dim 0 \
   --lora_alpha 0 \
   --lora_learning_rate 5e-4 \
   --lora_module_name layers. \
   --deepspeed \
   --output_dir $OUTPUT  \
   2>&1 | tee ${OUTPUT}/training.log
