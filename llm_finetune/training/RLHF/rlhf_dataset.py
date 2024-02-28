import sys
import os
import torch
from torch.utils.data import Dataset


class RLHFDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_prompt_len) -> None:
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
        self.max_prompt_len = max_prompt_len

    def __len__(self):
        length = len(self.raw_data_list)
        return length

    def __getitem__(self, idx):
        data = self.raw_data_list[idx]
        prompt = data["prompt"]

        input_ids = self.tokenizer.encode(text=prompt)
        if len(input_ids) > self.max_prompt_len:
            input_ids = input_ids[:self.max_prompt_len]
        attention_mask = [1 for ids in input_ids]

        return { "prompt" : input_ids,
                 "prompt_att_mask" : attention_mask, }


class DataCollatorRLHF:
    def __init__(self, max_prompt_len, pad_token_id):
        self.max_prompt_len = max_prompt_len
        self.pad_token_id = pad_token_id

    def __call__(self, data_list):
        batch = {}

        # left-padding
        for i, key in enumerate(["prompt", "prompt_att_mask"]):
            pad_value = self.pad_token_id if key == "prompt" else 0
            out_list = []
            for data in data_list:
                item = data[key]
                pad_len = self.max_prompt_len - len(item)
                if pad_len > 0:
                    item = [pad_value] * pad_len + item
                out_list.append(item)

            batch[key] = torch.tensor(out_list)

        return batch

