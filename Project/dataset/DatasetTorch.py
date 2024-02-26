# -*- coding:utf-8 -*-

from typing import Tuple
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from datasets.dataset_dict import DatasetDict


class DatasetMultiChoiceQA(Dataset):

    def __init__(
            self,
            dataset: DatasetDict,
            tokenizer: AutoTokenizer,
            splits: str = "train",
            ratio_range: Tuple[float, float] = (0.0, 1.0),
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

        self.ds = dataset[splits]  # of List[dict]

        self.len_full = len(self.ds)
        index_range = (max(0, int(self.len_full * ratio_range[0])),
                       min(self.len_full, int(self.len_full * ratio_range[1])))
        if 0 <= index_range[0] < index_range[1] <= self.len_full:
            self.ds = self.ds[index_range[0]: index_range[1]]  # of Dict[list]
        else:
            self.ds = self.ds[:]  # of Dict[list]
        # convert Dict[list] to List[dict]
        self.ds = [dict(zip(self.ds, v)) for v in zip(*self.ds.values())]

        self.ratio_range = ratio_range
        self.index_range = index_range

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self) -> int:
        return len(self.ds)
