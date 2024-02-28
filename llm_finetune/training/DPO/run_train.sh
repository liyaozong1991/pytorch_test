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

mkdir -p $OUTPUT


INPUT_DATA=""
EVAL_DATA=""


deepspeed --master_port 12346 main.py \
   --train_file "${INPUT_DATA}" \
   --eval_file "${EVAL_DATA}" \
   --model_name_or_path "${INPUT_MODEL}" \
   --per_device_train_batch_size 5\
   --per_device_eval_batch_size 6 \
   --max_seq_len 2048 \
   --learning_rate 1e-6 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --gradient_checkpointing \
   --offload_reference_model \
   --disable_dropout \
   --num_warmup_steps 100 \
   --only_optimize_lora \
   --lora_dim 0 \
   --lora_alpha 16 \
   --lora_learning_rate 5e-4 \
   --lora_module_name layers. \
   --deepspeed --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --beta 0.1 \
   --output_dir $OUTPUT  \
   2>&1 | tee ${OUTPUT}/training.log
