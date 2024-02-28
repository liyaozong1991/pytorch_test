#!/bin/bash

MODEL_PATH=$1
GPU=$2

if [ "$GPU" == "" ]; then
    GPU=0
fi


python generate.py \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${MODEL_PATH} \
    --interactive \
    --need_build_prompt \
    --gpus ${GPU} \
    #--only_cpu \
