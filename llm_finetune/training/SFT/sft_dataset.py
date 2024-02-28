
import sys
import os
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_seq_len) -> None:
        super().__init__()
        import json
        with open(json_file) as fin:
            if json_file.endswith(".json"):
                self.raw_data_list = json.load(fin)
            elif json_file.endswith(".jsonl"):
                self.raw_data_list = []
                for line in fin:
                    self.raw_data_list.append(json.loads(line))
            else:
                raise

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
        output = data["output"]

        p_ids = self.tokenizer.encode(text=prompt)
        all_ids = self.tokenizer.encode(text=prompt+output)

        input_ids = all_ids + [self.tokenizer.eos_token_id]
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        attention_mask = [1 for ids in input_ids]

        ## prompt部分不计算loss
        # NOTE prompt单独tokenize的结果可能与prompt+output再tokenize的结果在拼接处有出入
        context_length = min(len(p_ids), len(input_ids))
        labels = [-100] * context_length + input_ids[context_length:]

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.eos_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

