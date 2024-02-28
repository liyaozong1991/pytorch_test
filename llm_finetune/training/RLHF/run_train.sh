#!/bin/bash

ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
OUTPUT=$3
ACTOR_ZERO_STAGE=$4
CRITIC_ZERO_STAGE=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT


INPUT_DATA=""

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --master_port 12346 main.py \
   --train_file "${INPUT_DATA}" \
   --actor_model_name_or_path "$ACTOR_MODEL_PATH" \
   --critic_model_name_or_path "$CRITIC_MODEL_PATH" \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 1536 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 4 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 10 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --actor_loss_initial_scale_power 8 \
   --critic_loss_initial_scale_power 8 \
   --actor_lora_dim 0 \
   --actor_lora_alpha 16 \
   --actor_lora_learning_rate 5e-4 \
   --actor_lora_module_name model.layers. \
   --critic_lora_dim 0 \
   --critic_lora_alpha 16 \
   --critic_lora_learning_rate 5e-4 \
   --critic_lora_module_name model.layers. \
   --only_optimize_lora \
   --enable_adv_norm \
   --enable_reward_scaling \
   --kl_ctl 0.05 \
   --print_answers \
   --output_dir $OUTPUT \
   2>&1 | tee ${OUTPUT}/training.log

   #--enable_reward_norm \
   #--enable_hybrid_engine \
   #--align_overflow \
