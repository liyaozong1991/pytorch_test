import torch
import torch.distributed as dist
import torch.nn.functional as F
import sys
import os
import time

import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam

from transformers import AutoModelForCausalLM, get_scheduler

from typing import Dict, MutableMapping, Tuple, Union

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.ds_utils import get_train_ds_config, get_eval_ds_config
from utils.module.lora_peft import (convert_linear_layer_to_lora, only_optimize_lora_parameters,
                                    make_model_gradient_checkpointing_compatible, load_lora_weights,
                                    convert_linear_layer_to_lora_by_cfg, convert_lora_to_linear_layer)
from utils.model.model_utils import create_hf_model
from utils.utils import print_rank_0, get_optimizer_grouped_parameters


def log_init(model_name, stime=None):
    if torch.distributed.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DPOTrainer(object):
    def __init__(self, model_name_or_path, args, num_total_iters):
        self.args = args
        self.num_total_iters = num_total_iters
        self.model = self._init_model(model_name_or_path)
        self.ref_model = self._init_ref(model_name_or_path)
        self.ref_model.eval()

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # kl weight
        self.beta = args.beta

    def _init_model(self, model_name_or_path):
        stime = log_init("Policy")

        # DS Config
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            stage=self.args.zero_stage,
            enable_tensorboard=self.args.enable_tensorboard,
            tb_path=self.args.tensorboard_path,
            tb_name="dpo_model")
        ds_config['train_micro_batch_size_per_gpu'] = self.args.per_device_train_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        # Model
        model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=model_name_or_path,
            tokenizer=None,
            ds_config=ds_config,
            disable_dropout=self.args.disable_dropout)

        # LoRA
        if self.args.lora_dim > 0:
            model = convert_linear_layer_to_lora(model, self.args.lora_module_name,
                                                 lora_dim=self.args.lora_dim,
                                                 lora_alpha=self.args.lora_alpha)
            if self.args.load_lora_dir is not None:
                load_lora_weights(model, self.args.load_lora_dir, self.args.zero_stage)
            if self.args.only_optimize_lora:
                model = only_optimize_lora_parameters(model)
                model = make_model_gradient_checkpointing_compatible(model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            model, self.args.weight_decay,
            self.args.lora_learning_rate)
        optim = AdamOptimizer(optim_params,
                              lr=self.args.learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=optim,
                                          lr_scheduler=lr_scheduler,
                                          config=ds_config)

        log_init("Policy", stime=stime)
        return engine

    def _init_ref(self, model_name_or_path):
        stime = log_init("Ref")

        # DS Config
        zero_stage = self.args.zero_stage
        if zero_stage != 3:
            zero_stage = 0

        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       zero_stage)
        ds_config['train_micro_batch_size_per_gpu'] = self.args.per_device_train_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        ref_model = create_hf_model(AutoModelForCausalLM,
                                    model_name_or_path,
                                    None,
                                    ds_config)
        if self.args.load_lora_dir is not None:
            convert_linear_layer_to_lora_by_cfg(ref_model, self.args.load_lora_dir)
            load_lora_weights(ref_model, self.args.load_lora_dir, self.args.zero_stage)
            convert_lora_to_linear_layer(ref_model)

        ref_engine, *_ = deepspeed.initialize(model=ref_model,
                                              config=ds_config)

        log_init("Ref", stime=stime)
        return ref_engine

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
        self.ref_model.eval()

    def dpo_loss(self,
                 pi_logp_cho, pi_logp_rej,
                 ref_logp_cho, ref_logp_rej):
        pi_logp_ratio = pi_logp_cho - pi_logp_rej
        ref_logp_ratio = ref_logp_cho - ref_logp_rej
        loss = -F.logsigmoid(self.beta * (pi_logp_ratio - ref_logp_ratio))
        loss = loss.mean()
        return loss

    def run_dpo(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        output_mask = inputs["output_mask"]
        assert len(input_ids.shape) == 2
        assert (input_ids.shape[0] % 2) == 0

        action_mask = output_mask[:, 1:]

        # policy outputs
        pi_output = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
        pi_logits = pi_output.logits
        pi_logprob = gather_log_probs(pi_logits[:, :-1, :], input_ids[:, 1:])
        pi_logprob_sum = (pi_logprob * action_mask).sum(1)

        # ref outpus
        with torch.no_grad():
            ref_output = self.ref_model(input_ids, attention_mask=attention_mask, use_cache=False)
            ref_logits = ref_output.logits
            ref_logprob = gather_log_probs(ref_logits[:, :-1, :], input_ids[:, 1:])
            ref_logprob_sum = (ref_logprob * action_mask).sum(1)

        # split the logprob into two parts, chosen and rejected
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]
        pi_logp_cho = pi_logprob_sum[:bs]   # [bs,]
        pi_logp_rej = pi_logprob_sum[bs:]
        ref_logp_cho = ref_logprob_sum[:bs]
        ref_logp_rej = ref_logprob_sum[bs:]

        loss = self.dpo_loss(pi_logp_cho, pi_logp_rej, ref_logp_cho, ref_logp_rej)

        # for metric
        chosen_reward = self.beta * (pi_logp_cho - ref_logp_cho).detach()
        reject_reward = self.beta * (pi_logp_rej - ref_logp_rej).detach()

        return { "loss": loss,
                 "chosen_reward" : chosen_reward,
                 "reject_reward" : reject_reward, }

    def run_train_step(self, inputs):
        dpo_output = self.run_dpo(inputs)
        loss = dpo_output["loss"]
        self.model.backward(loss)
        self.model.step()
        return dpo_output

    def run_eval_step(self, inputs):
        with torch.no_grad():
            dpo_output = self.run_dpo(inputs)
        return dpo_output
