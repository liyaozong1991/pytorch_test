#!/bin/bash

MODEL_PATH=$1
INPUT_FILE=$2
OUTPUT_FILE=$3
GPU=$4


if [ "$GPU" == "" ]; then
    GPU=0
fi


python generate.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${MODEL_PATH} \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_FILE} \
    --batch_size 20 \
    --gpus ${GPU} \
