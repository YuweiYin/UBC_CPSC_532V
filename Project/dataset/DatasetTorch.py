# -*- coding:utf-8 -*-

from torch.utils.data import Dataset

from transformers import AutoTokenizer
from datasets.dataset_dict import DatasetDict


class DatasetMultiChoiceQA(Dataset):

    def __init__(
            self,
            dataset: DatasetDict,
            tokenizer: AutoTokenizer,
            splits: str = "train",
    ):
        super(DatasetMultiChoiceQA, self).__init__()

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
        #     "prompt": str,  # input prompt to Causal LMs and then output answer
        #   }

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.splits = splits

    def __getitem__(self, index):
        return self.dataset[self.splits][index]

    def __len__(self) -> int:
        return len(self.dataset[self.splits])
