import sys
import os
import torch
from torch.utils.data import Dataset


class RMDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_seq_len) -> None:
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

        chosen_ids = self.tokenizer.encode(text=prompt+chosen)
        reject_ids = self.tokenizer.encode(text=prompt+reject)

        def mod_ids(input_ids):
            input_ids = input_ids + [self.tokenizer.eos_token_id]
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]

            attention_mask = [1 for ids in input_ids]
            input_len = len(input_ids)
            pad_len = self.max_seq_len - input_len
            if pad_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

            return input_ids, attention_mask, input_len

        chosen_ids, chosen_attn_mask, chosen_len = mod_ids(chosen_ids)
        reject_ids, reject_attn_mask, reject_len = mod_ids(reject_ids)

        return { "input_ids" : [chosen_ids, reject_ids],
                 "attention_mask" : [chosen_attn_mask, reject_attn_mask],
                 "input_len" : [chosen_len, reject_len], }


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        for i, key in enumerate(["input_ids", "attention_mask", "input_len"]):
            batch[key] = torch.tensor(
                [ f[key][0] for f in data ] + [ f[key][1] for f in data ]
            )

        return batch

