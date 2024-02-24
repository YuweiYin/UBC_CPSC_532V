# -*- coding:utf-8 -*-
import os.path

import torch
from transformers import AutoTokenizer


class TokenizerLoader:

    def __init__(self):
        pass

    def get_tokenizer(
            self,
            model_name: str = "",
            local_path: str = "",
            cache_dir: str = "~/.cache/huggingface/datasets",
            padding_side: str = "right",
            truncation_side: str = "right",
    ):
        """
        Get the tokenizer via Hugging Face API. https://huggingface.co/docs/transformers/main_classes/tokenizer
        :param model_name: model name.
        :param local_path: file path of the local tokenizer.
        :param cache_dir: The directory where data & model are cached.
        :param padding_side: padding_side of the tokenizer ("right" for training, "left" for testing).
        :param truncation_side: truncation_side of the tokenizer ("right" for training, "left" for testing).
        :return: the tokenizer.
        """

        if isinstance(local_path, str) and os.path.isfile(local_path):
            try:
                print(f"Loading local tokenizer from: {local_path}")  # TODO: test loading local tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    local_path, cache_dir=cache_dir, padding_side=padding_side, truncation_side=truncation_side)
                return tokenizer
            except Exception as e:
                print(e)

        if not isinstance(model_name, str) or model_name == "":
            raise ValueError(f"[{self.__class__.__name__}] ValueError: model_name = {model_name}")

        model_name = model_name.strip()
        print(f"Loading tokenizer from Hugging Face. model_name: {model_name}")

        if model_name == "gpt1":
            # openai-gpt (a.k.a. "GPT-1") is the first transformer-based language model created and released by OpenAI
            tokenizer = AutoTokenizer.from_pretrained(
                "openai-community/openai-gpt", cache_dir=cache_dir,
                padding_side=padding_side, truncation_side=truncation_side)
        elif model_name == "gpt2":
            # The smallest version of GPT-2, with 124M parameters.
            tokenizer = AutoTokenizer.from_pretrained(
                "openai-community/gpt2", cache_dir=cache_dir,
                padding_side=padding_side, truncation_side=truncation_side)
        elif model_name == "gpt2-medium":
            # GPT-2 Medium is the 355M parameter version of GPT-2
            tokenizer = AutoTokenizer.from_pretrained(
                "openai-community/gpt2-medium", cache_dir=cache_dir,
                padding_side=padding_side, truncation_side=truncation_side)
        elif model_name == "gpt2-large":
            # GPT-2 Large is the 774M parameter version of GPT-2
            tokenizer = AutoTokenizer.from_pretrained(
                "openai-community/gpt2-large", cache_dir=cache_dir,
                padding_side=padding_side, truncation_side=truncation_side)
        elif model_name == "gpt2-xl":
            # GPT-2 XL is the 1.5B parameter version of GPT-2
            tokenizer = AutoTokenizer.from_pretrained(
                "openai-community/gpt2-xl", cache_dir=cache_dir,
                padding_side=padding_side, truncation_side=truncation_side)
        else:
            raise ValueError(f"[DataLoader.get_splits] ValueError: ds_name = {model_name}")

        return tokenizer
