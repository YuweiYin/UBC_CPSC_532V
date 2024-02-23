# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
# from datasets import arrow_dataset
from datasets.dataset_dict import DatasetDict


class DatasetMultiChoiceQA(Dataset):

    def __init__(
            self,
            dataset: DatasetDict,
            tokenizer: AutoTokenizer,
            splits: str = "train",
    ):
        super(DatasetMultiChoiceQA, self).__init__()

        # ord_A = ord("A")
        # self.idx2choice = {index: chr(ord_A + index) for index in range(26)}
        # self.choice2idx = {chr(ord_A + index): index for index in range(26)}

        # Unified data item format:
        #   {
        #     "dataset": str,
        #     "id": Optional[str],
        #     "question": str,
        #     "context": Optional[str],
        #     "concept": Optional[List[str]],
        #     "choices_label": List[str], (e.g., ["A", "B", "C", "D"])
        #     "choices_text": List[str],
        #     "answer": List[str], (e.g., ["A"] or ["A", "C"])
        #     "prompt": str,  # input prompt to Causal LMs and then output answer_label
        #   }

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.splits = splits

    def preprocessing_train(self, example):
        input_ids = []
        # labels = []

        # value_list = example["data"]  # all human/assistant dialogue
        value_list = example["data"][:2]  # only the first round of the human/assistant dialogue
        value_ids_list = [self.tokenizer.encode(value) for value in value_list]
        for idx, value_ids in enumerate(value_ids_list):
            if idx % 2 == 0:  # human
                input_ids += self.user_tokens + value_ids
                # labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            else:  # assistant
                input_ids += self.assistant_tokens + value_ids
                # labels += [self.ignore_index] + value_ids

        input_ids.append(self.tokenizer.eos_token_id)
        input_ids = input_ids[: self.model_max_length]  # trimming
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))  # right padding
        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def preprocessing_eval(self, example):
        input_ids = []
        labels = []

        # value_list = example["data"]  # all human/assistant dialogue
        value_list = example["data"][:2]  # only the first round of the human/assistant dialogue
        value_ids_list = [self.tokenizer.encode(value) for value in value_list]
        for idx, value_ids in enumerate(value_ids_list):
            if idx % 2 == 0:  # human
                # input_ids += self.user_tokens + value_ids
                input_ids += self.user_tokens + value_ids + self.assistant_tokens
            else:  # assistant
                # labels += self.assistant_tokens + value_ids
                labels += value_ids + [self.tokenizer.eos_token_id]
                break

        input_ids = input_ids[: self.model_max_length]  # trimming
        # input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))  # right padding
        input_ids = [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids)) + input_ids  # left padding
        input_ids = torch.LongTensor(input_ids)
        # labels = input_ids.clone()
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, index):
        return self.dataset[self.splits][index]
        # if self.splits == "train":
        #     return self.preprocessing_train(self.dataset[self.splits][index])
        # else:
        #     return self.preprocessing_eval(self.dataset[self.splits][index])

    def __len__(self) -> int:
        return len(self.dataset[self.splits])
