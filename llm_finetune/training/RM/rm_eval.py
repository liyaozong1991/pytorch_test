#!/usr/bin/env python

import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from rm_dataset import RMDataset, DataCollatorReward

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.utils import print_rank_0, to_device, set_random_seed, get_all_reduce_mean
from utils.ds_utils import get_eval_ds_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
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
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step2_tensorboard")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

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


    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = 0

    eval_dataset = RMDataset(args.eval_file, tokenizer, args.max_seq_len)
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        eval_sampler = DistributedSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    # Model
    zero_stage = args.zero_stage
    if zero_stage != 3:
        zero_stage = 0
    ds_config = get_eval_ds_config(offload=args.offload,
                                   stage=zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_eval_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_eval_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    ds_eval_config = get_eval_ds_config(offload=False,
                                        stage=zero_stage)

    # We need to set train batch size and micro batch size here to pass the sanity check of DeepSpeed engine.
    ds_eval_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_eval_batch_size
    ds_eval_config[
        'train_batch_size'] = args.per_device_eval_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    reward_model = create_critic_model(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        ds_config=ds_eval_config,
        load_from_ckpt=True,
        disable_dropout=args.disable_dropout,
        zero_stage=zero_stage)

    reward_engine, *_ = deepspeed.initialize(model=reward_model,
                                             config=ds_config)

    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        chosen_scores = 0
        reject_scores = 0
        score_diffs = 0

        pbar = tqdm(total=len(eval_dataloader),
                    ascii=True,
                    disable=(args.global_rank!=0))

        all_score_list = []
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            cur_pred_num = chosen.shape[0]

            correct_predictions += (chosen > rejected).sum()
            total_predictions += cur_pred_num
            chosen_scores += chosen.mean().float()
            reject_scores += rejected.mean().float()
            score_diffs += (chosen - rejected).mean().float()

            for i in range(cur_pred_num):
                all_score_list.append(chosen[i].item())
                all_score_list.append(rejected[i].item())

            pbar.update(1)
        pbar.close()

        acc = correct_predictions / total_predictions
        chosen_scores = chosen_scores / (step + 1)
        reject_scores = reject_scores / (step + 1)
        score_diffs = score_diffs / (step + 1)
        try:
            acc = get_all_reduce_mean(acc).item()
            chosen_scores = get_all_reduce_mean(chosen_scores).item()
            reject_scores = get_all_reduce_mean(reject_scores).item()
            score_diffs = get_all_reduce_mean(score_diffs).item()
        except:
            pass

        # total mean and var
        all_score = sum(all_score_list)
        sum_and_count = torch.tensor([all_score, len(all_score_list)], device=device)
        dist.all_reduce(sum_and_count, dist.ReduceOp.SUM)
        global_sum, count = sum_and_count
        global_mean = (global_sum / count).item()
        sum_var = sum([(score - global_mean) ** 2 for score in all_score_list])
        sum_var = torch.tensor(sum_var, device=device)
        dist.all_reduce(sum_var, dist.ReduceOp.SUM)
        global_var = (sum_var / count).item()

        return chosen_scores, reject_scores, score_diffs, acc, global_mean, global_var

    print_rank_0("***** Running evaluating *****", args.global_rank)
    print_rank_0(f"Batch num per device = {len(eval_dataloader)}", args.global_rank)
    chosen_scores, reject_scores, score_diffs, acc, global_mean, global_var = \
            evaluation_reward(reward_engine, eval_dataloader)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {chosen_scores},\n"
        f"reject_last_scores (lower is better) : {reject_scores},\n"
        f"score_diffs (bigger is better) : {score_diffs},\n"
        f"acc (higher is better) : {acc}\n"
        f"global mean : {global_mean}\n"
        f"global var : {global_var}",
        args.global_rank)


if __name__ == "__main__":
    main()
