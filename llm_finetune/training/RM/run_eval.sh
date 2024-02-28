#!/bin/bash

INPUT_MODEL=$1
ZERO_STAGE=$2

if [ r"$ZERO_STAGE" != r"3" ]; then
    ZERO_STAGE=0
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


EVAL_DATA=""


deepspeed rm_eval.py \
   --eval_file "${EVAL_DATA}" \
   --model_name_or_path "${INPUT_MODEL}" \
   --per_device_eval_batch_size 16 \
   --max_seq_len 2048 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   2>&1 | tee ${INPUT_MODEL}/eval.log
