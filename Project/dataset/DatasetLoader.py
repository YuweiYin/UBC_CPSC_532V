# -*- coding:utf-8 -*-

from typing import Tuple, Optional

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict


class DatasetLoader:

    def __init__(self):
        # ord_A = ord("A")
        # self.idx2choice = {index: chr(ord_A + index) for index in range(26)}
        # self.choice2idx = {chr(ord_A + index): index for index in range(26)}
        pass

    def get_dataset(
            self,
            ds_name: str = "",
            cache_dir: Optional[str] = None,  # "~/.cache/huggingface/"
    ) -> Tuple[Optional[DatasetDict], bool]:
        """
        Get the dataset via Hugging Face API. https://huggingface.co/datasets
        :param ds_name: dataset name.
        :param cache_dir: The directory where data & model are cached.
        :return: dataset and test_label (whether the test set has label)
        """

        if not isinstance(ds_name, str) or ds_name == "":
            raise ValueError(f"[{self.__class__.__name__}] ValueError: ds_name = {ds_name}")

        ds_name = ds_name.strip()

        test_label = False

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

        if ds_name == "commonsense_qa":
            dataset = load_dataset("tau/commonsense_qa", cache_dir=cache_dir)
            """
            original_example = {
                'id': '075e483d21c29a511267ef62bedc0461',
                'question': 'The sanctions against the school were a punishing blow, and they seemed to what the '
                            'efforts the school had made to change?',
                'question_concept': 'punishing',
                'choices': {
                    'label': ['A', 'B', 'C', 'D', 'E'],
                    'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']
                },
                'answerKey': 'A'
            }
            """
            # rename columns
            dataset = dataset.rename_columns({
                "question_concept": "concept",
                "answerKey": "answer",
            })
            # add columns (map)
            # dataset = dataset.map(lambda item, idx: {"dataset": "Commonsense QA"}, with_indices=True)
            dataset = dataset.map(lambda item, idx: {"dataset": "Commonsense Question Answering"}, with_indices=True)
            dataset = dataset.map(lambda item, idx: {"context": ""}, with_indices=True)
            dataset = dataset.map(lambda item, idx: {"concept": [item["concept"]]}, with_indices=True)
            dataset = dataset.map(lambda item, idx: {"choices_label": item["choices"]["label"]}, with_indices=True)
            dataset = dataset.map(lambda item, idx: {"choices_text": item["choices"]["text"]}, with_indices=True)
            dataset = dataset.map(self.map_prompt, with_indices=True)
            # remove columns
            dataset = dataset.remove_columns([
                "choices",
            ])
        else:
            raise ValueError(f"[DataLoader.get_splits] ValueError: ds_name = {ds_name}")

        return dataset, test_label

    @staticmethod
    def map_prompt(item, idx: int = 0):
        # prompt = f"Answer the multiple-choice question of the {_item['dataset']} task.\n"  # instruction
        prompt = f"Task: {item['dataset']}.\n"
        if "concept" in item and isinstance(item["concept"], list) and len(item["concept"]) > 0:
            # prompt += f"The question is related to the following concepts: {', '.join(_item['concept'])}.\n"
            prompt += f"Concepts: {', '.join(item['concept'])}.\n"
        if "context" in item and isinstance(item["context"], str) and len(item["context"]) > 0:
            prompt += f"Context: {item['context']}\n"
        prompt += f"Question: {item['question']}\n"

        assert len(item["choices_label"]) == len(item["choices_text"])
        for cl, ct in zip(item["choices_label"], item["choices_text"]):
            prompt += f"{cl}: {ct}\n"

        # prompt += f"Answer: {_item['answer_label']}\n"

        item["prompt"] = prompt

        return item
