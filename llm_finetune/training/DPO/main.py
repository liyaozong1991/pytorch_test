#!/usr/bin/env python

import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
)

import deepspeed

from dpo_trainer import DPOTrainer
from dpo_dataset import DPODataset, DataCollatorDPO

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, save_zero_three_model, load_hf_tokenizer
from utils.module.lora_peft import convert_lora_to_linear_layer, save_lora


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        '--train_file',
        type=str,
    )
    parser.add_argument(
        '--eval_file',
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # DPO
    parser.add_argument("--beta",
                        type=float,
                        default=0.1,
                        help="Weight of KL pernalty")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--offload_reference_model',
                        action='store_true',
                        help='Enable Offload for reference model.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_alpha",
                        type=float,
                        default=1,
                        help="LoRA scaling factor, scaling = lora_alpha / lora_dim")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--load_lora_dir",
        type=str,
        help="Path to pretrained LoRA parameter ckpt.",
    )
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="dpo_tensorboard")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    ## Validate settings
    # LoRA
    if args.load_lora_dir is not None:
        assert args.lora_dim > 0, \
            "LoRA training should be enabled for loading pretrained lora "\
            "(Unless you know what it will affect)"

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, pad_token_id=0)

    data_collator = DataCollatorDPO()
    train_dataset = DPODataset(args.train_file, tokenizer, args.max_seq_len)
    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  num_workers=1,
                                  batch_size=args.per_device_train_batch_size)

    if args.eval_file:
        eval_dataset = DPODataset(args.eval_file, tokenizer, args.max_seq_len)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_dataset)
        else:
            eval_sampler = DistributedSampler(eval_dataset)

        eval_dataloader = DataLoader(eval_dataset,
                                     collate_fn=data_collator,
                                     sampler=eval_sampler,
                                     batch_size=args.per_device_eval_batch_size)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    num_total_iters = args.num_train_epochs * num_update_steps_per_epoch

    # Init DPOTrainer
    dpo_trainer = DPOTrainer(args.model_name_or_path, args, num_total_iters)

    def evaluation(dpo_trainer, eval_dataloader):
        dpo_trainer.eval()
        correct_predictions = 0
        total_predictions = 0
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            outputs = dpo_trainer.run_eval_step(batch)

            loss = outputs["loss"]
            chosen_reward = outputs["chosen_reward"]
            reject_reward = outputs["reject_reward"]

            correct_predictions += (chosen_reward > reject_reward).sum()
            total_predictions += chosen_reward.shape[0]
            losses += loss.float()

            if step >= 30:  # For faster evaluation and debugging
                break

        acc = correct_predictions / total_predictions
        losses = losses / (step + 1)
        try:
            acc = get_all_reduce_mean(acc).item()
            losses = get_all_reduce_mean(losses).item()
        except:
            pass
        return losses, acc


    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    if args.eval_file:
        print_rank_0(
            f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
            args.global_rank)
        loss, acc = evaluation(dpo_trainer, eval_dataloader)
        print_rank_0(
            f"loss: {loss}, acc (higher is better) : {acc}",
            args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)

        dpo_trainer.train()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = dpo_trainer.run_train_step(batch)
            loss = outputs["loss"]
            mean_loss += loss.item()

            if (step % 10) == 0 and step > 0:
                print_rank_0(f"Current loss={loss.item()}, mean loss={mean_loss/step}", args.global_rank)

        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)

        if args.eval_file:
            print_rank_0(
                f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
                args.global_rank)
            loss, acc = evaluation(dpo_trainer, eval_dataloader)
            print_rank_0(
                f"loss: {loss}, acc (higher is better) : {acc}",
                args.global_rank)

        dpo_trainer.model.tput_timer.update_epoch_count()


    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)

        if args.lora_dim > 0:
            save_lora(dpo_trainer.model.module, args.output_dir, args.zero_stage)
        dpo_trainer.model = convert_lora_to_linear_layer(dpo_trainer.model)

        if torch.distributed.get_rank() == 0:
            save_hf_format(dpo_trainer.model, tokenizer, args)

        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(dpo_trainer.model,
                                  global_rank=args.global_rank,
                                  save_dir=args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
