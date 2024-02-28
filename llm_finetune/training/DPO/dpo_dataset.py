import sys
import os
import torch
from torch.utils.data import Dataset



class DPODataset(Dataset):
    def __init__(self, json_file, tokenizer, max_seq_len=None) -> None:
        super().__init__()
        import json
        with open(json_file) as fin:
            if json_file.endswith(".json"):
                self.raw_data_list = json.load(fin)
            else:
                self.raw_data_list = []
                for line in fin:
                    self.raw_data_list.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        length = len(self.raw_data_list)
        return length

    def __getitem__(self, idx):
        return self._get_item(idx)

    def _get_item(self, idx):
        data = self.raw_data_list[idx]

        prompt = data["prompt"]
        chosen = data["chosen"]
        reject = data["reject"]

        prompt_ids = self.tokenizer.encode(text=prompt)
        all_chosen_ids = self.tokenizer.encode(text=prompt+chosen)
        all_reject_ids = self.tokenizer.encode(text=prompt+reject)

        def mod_ids(prompt_ids, all_ids):
            input_ids = all_ids + [self.tokenizer.eos_token_id]

            output_mask = [1 for ids in input_ids]
            # prompt部分不计算loss
            # NOTE prompt单独tokenize的结果可能与prompt+output再tokenize的结果在拼接处有出入
            output_mask[:len(prompt_ids)] = 0
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                output_mask = output_mask[:self.max_seq_len]

            attention_mask = [1 for ids in input_ids]
            input_len = len(input_ids)
            pad_len = self.max_seq_len - input_len
            if pad_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                output_mask = output_mask + [0] * pad_len
                attention_mask = attention_mask + [0] * pad_len

            return input_ids, attention_mask, output_mask

        chosen_ids, chosen_attn_mask, chosen_out_mask = mod_ids(prompt_ids, all_chosen_ids)
        reject_ids, reject_attn_mask, reject_out_mask = mod_ids(prompt_ids, all_reject_ids)

        return { "input_ids" : [chosen_ids, reject_ids],
                 "attention_mask" : [chosen_attn_mask, reject_attn_mask],
                 "output_mask" : [chosen_out_mask, reject_out_mask], }


class DataCollatorDPO:

    def __call__(self, data):
        batch = {}
        for i, key in enumerate(["input_ids", "attention_mask", "output_mask"]):
            batch[key] = torch.tensor(
                [ f[key][0] for f in data ] + [ f[key][1] for f in data ]
            )

        return batch

