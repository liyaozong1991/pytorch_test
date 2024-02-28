import math
import os
import time
import json

from typing import Union
from collections.abc import Iterable

import deepspeed
import torch
import torch.nn.functional as F
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
from torch import nn

from utils.utils import load_state_dict_into_model


LORA_CFG_NAME = "adapter_config.json"
LORA_CKPT_NAME = "adapter_model.bin"


class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_alpha=1,
                 lora_dropout=0,
                 bias=None):
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            out_features, in_features = weight.ds_shape
        except:
            out_features, in_features = weight.shape
        self.lora_A = nn.Linear(in_features, lora_dim, bias=False)
        self.lora_B = nn.Linear(lora_dim, out_features, bias=False)
        self.lora_scaling = lora_alpha / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def get_full_lora_weights(self):
        full_lora = self.lora_B.weight @ self.lora_A.weight * self.lora_scaling
        return full_lora

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            full_lora = self.get_full_lora_weights()
            self.weight.data += full_lora
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            full_lora = self.get_full_lora_weights()
            self.weight.data -= full_lora
        self.fuse_lora = False

    def forward(self, input):
        base_result = F.linear(input, self.weight, self.bias)
        if self.fuse_lora:
            return base_result
        else:
            lora_result = self.lora_B(self.lora_A(self.lora_dropout(input)))
            return base_result + self.lora_scaling * lora_result


def _get_replace_name(model: nn.Module,
                      target_module_name: Union[Iterable[str], str]) -> list[str]:
    is_single_pattern = isinstance(target_module_name, str)

    replace_name = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if is_single_pattern:
            target_module_found = target_module_name in name
        else:
            target_module_found = name in target_module_name or any(
                name.endswith(f".{target_key}") for target_key in target_module_name
            )

        if target_module_found:
            replace_name.append(name)
    return replace_name


def _to_lora_linear(model: nn.Module,
                    replace_name: Iterable[str],
                    lora_dim: int,
                    lora_alpha: int,
                    lora_dropout: int = 0) -> nn.Module:
    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(
            module.weight, lora_dim, lora_alpha, lora_dropout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model


# convert the linear layer to LoRA
def convert_linear_layer_to_lora(model,
                                 part_module_name,
                                 lora_dim=0,
                                 lora_alpha=1,
                                 lora_dropout=0):
    replace_name = _get_replace_name(model, part_module_name)
    model = _to_lora_linear(model, replace_name, lora_dim, lora_alpha, lora_dropout)

    # prepare peft format config
    peft_lora_cfg = {
        "base_model_name_or_path": model.config._name_or_path,
        "bias": "none",
        "r": lora_dim,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "peft_type": "LORA",
        "target_modules": list(sorted(set(replace_name))),
        "task_type": "CAUSAL_LM",
    }
    model._peft_lora_cfg = peft_lora_cfg

    return model

# convert the linear layer to LoRA by config file
def convert_linear_layer_to_lora_by_cfg(model: nn.Module,
                                        lora_ckpt_path: str) -> nn.Module:
    model_cfg_path = os.path.join(lora_ckpt_path, LORA_CFG_NAME)
    assert os.path.exists(
        model_cfg_path
    ), f"Cannot find model config file at {model_cfg_path}"
    with open(model_cfg_path, "r") as fin:
        cfg = json.load(fin)
    lora_dim = cfg["r"]
    lora_alpha = cfg["lora_alpha"]
    lora_dropout = cfg["lora_dropout"]
    target_modules = cfg["target_modules"]

    replace_name = _get_replace_name(model, target_modules)
    model = _to_lora_linear(model, replace_name, lora_dim, lora_alpha, lora_dropout)
    model._peft_lora_cfg = cfg

    return model


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
        partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]


# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        params_to_fetch = _z3_params_to_fetch(
            [module.weight,
             module.bias,
             module.lora_A.weight,
             module.lora_B.weight, ]
        )
        with deepspeed.zero.GatheredParameters(params_to_fetch,
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_lora_weight()
    return model


# load LoRA weights from PEFT-format-LoRA-checkpoint
def load_lora_weights(model, lora_ckpt_path, zero_stage=0):
    model_ckpt_path = os.path.join(lora_ckpt_path, LORA_CKPT_NAME)
    assert os.path.exists(
        model_ckpt_path
    ), f"Cannot find model checkpoint at {model_ckpt_path}"
    lora_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')

    # remove peft save prefix
    input_state_dict = {}
    for k, v in lora_ckpt_state_dict.items():
        input_state_dict[k.removeprefix('base_model.model.')] = v

    # check LoRA params in state_dict
    to_load_name = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            to_load_name.append(name)
    will_load_any = False
    cant_load_name = []
    for lora_name in to_load_name:
        if lora_name not in input_state_dict:
            cant_load_name.append(lora_name)
        else:
            will_load_any = True
    assert will_load_any is True, \
        f"Cannot find any suitable LoRA params in {model_ckpt_path}"
    if torch.distributed.get_rank() == 0:
        if len(cant_load_name) > 0:
            print(f"> These LoRA params won't load from {model_ckpt_path}:")
            for lora_name in cant_load_name:
                print(f"{lora_name}")
            print(f"> ====================================================")

    start = time.time()
    load_state_dict_into_model(model,
                               input_state_dict,
                               "",
                               zero_stage=zero_stage)
    end = time.time()
    if torch.distributed.get_rank() == 0:
        print(f"> Loading LoRA state dict took {end - start} seconds")
    return model


def save_lora(model, lora_ckpt_path, zero_stage=0):
    # save peft format config file
    os.makedirs(lora_ckpt_path, exist_ok=True)
    output_cfg_file = os.path.join(lora_ckpt_path, LORA_CFG_NAME)
    cfg = model._peft_lora_cfg
    if torch.distributed.get_rank() == 0:
        with open(output_cfg_file, "w") as fout:
            json.dump(cfg, fout, ensure_ascii=False, indent=2)

    # save peft format weights
    save_lora_weights(model, lora_ckpt_path, zero_stage)


# save LoRA weights to PEFT-format-LoRA-checkpoint
def save_lora_weights(model, lora_ckpt_path, zero_stage=0):
    global_rank = torch.distributed.get_rank()
    # wrap for peft format
    model_to_save = nn.ModuleDict(
        {
            'base_model': nn.ModuleDict(
                {
                    'model': model,
                }
            )
        }
    )

    zero_stage_3 = (zero_stage == 3)
    os.makedirs(lora_ckpt_path, exist_ok=True)
    output_model_file = os.path.join(lora_ckpt_path, LORA_CKPT_NAME)

    if not zero_stage_3:
        if global_rank == 0:
            save_dict = model_to_save.state_dict()
            for key in list(save_dict.keys()):
                if "lora" not in key:
                    del save_dict[key]
            torch.save(save_dict, output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if "lora" not in k:
                continue
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def make_model_gradient_checkpointing_compatible(model):
    # Higgingface added this enable input require grads function to make gradient checkpointing work for lora-only optimization
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model
