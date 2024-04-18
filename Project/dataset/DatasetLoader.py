# -*- coding:utf-8 -*-

import random
import logging
from typing import Tuple, Optional

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict


class DatasetLoader:

    def __init__(self, logger: logging.Logger = None):
        # ord_A = ord("A")
        # self.idx2choice = {index: chr(ord_A + index) for index in range(26)}
        # self.choice2idx = {chr(ord_A + index): index for index in range(26)}
        if isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            logging.basicConfig(
                format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
            )
            self.logger = logging.getLogger(self.__class__.__name__)

    def load_dataset(
            self,
            ds_name: str,
            n_icl: int = 5,
            cache_dir: Optional[str] = None,  # "~/.cache/huggingface/"
            random_seed: int = 42,
            verbose: bool = False,
    ) -> dict:
        """
        Load the dataset via Hugging Face API. https://huggingface.co/datasets
        :param ds_name: dataset name.
        :param n_icl: The number of examples for in-context learning.
        :param cache_dir: The directory where data & model are cached.
        :param random_seed: Random seed of all modules.
        :param verbose: Verbose model: show logs.
        :return: the dataset dict.
        """

        dataset_hf, test_label = self.get_dataset(ds_name=ds_name, cache_dir=cache_dir)

        # Dataset split (training/validation/test sets)
        dataset_hf = dataset_hf.shuffle(seeds=random_seed)
        ds_hf_train, ds_hf_valid = dataset_hf["train"], dataset_hf["validation"]
        if test_label:
            ds_hf_test = dataset_hf["test"]
        else:  # split half of the validation set as the test set
            dataset_split = ds_hf_valid.train_test_split(test_size=0.5, shuffle=False)
            ds_hf_valid = dataset_split["train"]
            ds_hf_test = dataset_split["test"]
            dataset_hf["validation"] = ds_hf_valid
            dataset_hf["test"] = ds_hf_test
        del dataset_split

        # Show dataset information
        if verbose:
            self.logger.info(f"[Dataset] Training set shape: {ds_hf_train.shape}")
            self.logger.info(f"[Dataset] Validation set shape: {ds_hf_valid.shape}")
            self.logger.info(f"[Dataset] Test set shape: {ds_hf_test.shape}")
            assert ds_hf_train.column_names == ds_hf_valid.column_names == ds_hf_test.column_names, \
                "Assertion Error: column_names mismatch"
            self.logger.info(f"[Dataset] column names: {ds_hf_train.column_names}")
            self.logger.info(f"[Dataset] features: {ds_hf_train.features}\n")

        # Set in-context learning examples (random choice at least 3 examples from the training set)
        icl_prompt = self.get_icl_prompts(source_dataset=ds_hf_train, n_icl=n_icl, verbose=verbose)

        return {
            "dataset_hf": dataset_hf,
            "icl_prompt": icl_prompt,
        }

    def get_icl_prompts(
            self,
            source_dataset,
            n_icl: int = 5,
            verbose: bool = False,
    ):
        """
        Set in-context learning examples (random choice at least 3 examples from the source dataset)
        :param source_dataset: The dataset where the in-context-learning prompts come from.
        :param n_icl: The number of examples for in-context learning.
        :param verbose: Verbose model: show logs.
        :return: the in-context-learning prompts.
        """

        icl_indices = random.sample(range(len(source_dataset)), max(0, n_icl))
        icl_prompt = ""
        if len(icl_indices) > 0:
            icl_dataset = source_dataset.select(icl_indices)
            for icl_item in icl_dataset:
                icl_item = self.map_prompt(icl_item)  # get the prompt (without answer)
                # cur_prompt = icl_item["prompt"] + f"Answer: {icl_item['answer']}\n\n"  # set answers for ICL examples
                cur_prompt = icl_item["prompt"] + f"Answer:{icl_item['answer']}\n\n"  # set answers for ICL examples
                icl_prompt += cur_prompt
            # icl_prompt_len = len(icl_prompt)
            if verbose:
                self.logger.info(f"[Prompt] In-context Learning ({n_icl} examples):\n{icl_prompt}")
        else:
            if verbose:
                self.logger.info(f"[Prompt] NOT use In-context Learning")

        return icl_prompt

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
