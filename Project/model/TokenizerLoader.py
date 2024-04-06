# -*- coding:utf-8 -*-

import os
import logging
from typing import Optional

from transformers import AutoTokenizer
import hf_olmo


class TokenizerLoader:

    def __init__(self, logger: logging.Logger = None):
        if isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            logging.basicConfig(
                format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
            )
            self.logger = logging.getLogger(self.__class__.__name__)

    def load_tokenizer(
            self,
            model_name: str,
            is_train: bool = True,
            cache_dir: Optional[str] = None,  # "~/.cache/huggingface/"
            verbose: bool = False,
    ) -> dict:
        """
        Get the tokenizer via Hugging Face API. https://huggingface.co/docs/transformers/main_classes/tokenizer
        :param model_name: model name.
        :param is_train: Whether the tokenizer is for training of not.
        :param cache_dir: The directory where data & model are cached.
        :param verbose: Verbose model: print logs.
        :return: the tokenizer dict.
        """

        if is_train:
            tokenizer = self.get_tokenizer(
                model_name=model_name, cache_dir=cache_dir, padding_side="right", truncation_side="right")
            # tokenizer = self.get_tokenizer(
            #     model_name=model_name, cache_dir=None, padding_side="right", truncation_side="right")
        else:
            tokenizer = self.get_tokenizer(
                model_name=model_name, cache_dir=cache_dir, padding_side="left", truncation_side="left")
            # tokenizer = self.get_tokenizer(
            #     model_name=model_name, cache_dir=None, padding_side="left", truncation_side="left")

        # Special tokens
        # pad_token = "<|padoftext|>"
        # tokenizer.add_tokens([pad_token], special_tokens=True)
        # tokenizer.pad_token = pad_token
        # tokenizer.pad_token = tokenizer.bos_token
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_tokens(["<bos>", "<eos>", "<unk>", "<pad>"], special_tokens=True)
        # tokenizer.bos_token = "<bos>"
        # tokenizer.eos_token = "<eos>"
        # tokenizer.unk_token = "<unk>"
        # tokenizer.pad_token = "<pad>"
        # tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        # tokenizer.add_special_tokens({"bos_token": "[BOS]"})

        # Show tokenizer information
        if verbose:
            self.logger.info(f"[Tokenizer] (is_train: {is_train}) vocab size: {tokenizer.vocab_size}")
            self.logger.info(f"[Tokenizer] (is_train: {is_train}) all special tokens: "
                             f"{tokenizer.all_special_tokens}\n")

        return {
            "tokenizer": tokenizer,
        }

    def get_tokenizer(
            self,
            model_name: str = "",
            local_dir: str = "",
            cache_dir: Optional[str] = None,  # "~/.cache/huggingface/"
            padding_side: str = "right",
            truncation_side: str = "right",
    ):
        """
        Get the tokenizer via Hugging Face API. https://huggingface.co/docs/transformers/main_classes/tokenizer
        :param model_name: model name.
        :param local_dir: the directory of the local tokenizer (from tokenizer.save_pretrained(local_dir)).
        :param cache_dir: The directory where data & model are cached.
        :param padding_side: padding_side of the tokenizer ("right" for training, "left" for testing).
        :param truncation_side: truncation_side of the tokenizer ("right" for training, "left" for testing).
        :return: the tokenizer.
        """

        if isinstance(local_dir, str) and os.path.isdir(local_dir):
            # Try to load the local tokenizer first
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_dir)
                return tokenizer
            except Exception as e:
                self.logger.info(f"[get_tokenizer] >>> local_dir not effective:\n{e}")

        if not isinstance(model_name, str) or model_name == "":
            raise ValueError(f"[{self.__class__.__name__}] ValueError: model_name = {model_name}")

        model_name = model_name.strip()
        self.logger.info(f"Loading tokenizer from Hugging Face. model_name: {model_name}")

        match model_name:
            case "gpt1":
                # openai-gpt ("GPT-1"): the first transformer-based language model created and released by OpenAI
                tokenizer = AutoTokenizer.from_pretrained(
                    "openai-community/openai-gpt", cache_dir=cache_dir,
                    padding_side=padding_side, truncation_side=truncation_side)
            case "gpt2":
                # The smallest version of GPT-2, with 124M parameters.
                tokenizer = AutoTokenizer.from_pretrained(
                    "openai-community/gpt2", cache_dir=cache_dir,
                    padding_side=padding_side, truncation_side=truncation_side)
            case "gpt2-medium":
                # GPT-2 Medium is the 355M parameter version of GPT-2
                tokenizer = AutoTokenizer.from_pretrained(
                    "openai-community/gpt2-medium", cache_dir=cache_dir,
                    padding_side=padding_side, truncation_side=truncation_side)
            case "gpt2-large":
                # GPT-2 Large is the 774M parameter version of GPT-2
                tokenizer = AutoTokenizer.from_pretrained(
                    "openai-community/gpt2-large", cache_dir=cache_dir,
                    padding_side=padding_side, truncation_side=truncation_side)
            case "gpt2-xl":
                # GPT-2 XL is the 1.5B parameter version of GPT-2
                tokenizer = AutoTokenizer.from_pretrained(
                    "openai-community/gpt2-xl", cache_dir=cache_dir,
                    padding_side=padding_side, truncation_side=truncation_side)
            case "olmo-1b":
                # https://huggingface.co/allenai/OLMo-1B
                tokenizer = AutoTokenizer.from_pretrained(
                    "allenai/OLMo-1B", cache_dir=cache_dir,
                    padding_side=padding_side, truncation_side=truncation_side)
            case "olmo-7b":
                # https://huggingface.co/allenai/OLMo-7B
                tokenizer = AutoTokenizer.from_pretrained(
                    "allenai/OLMo-7B", cache_dir=cache_dir,
                    padding_side=padding_side, truncation_side=truncation_side)
            case "mistral-7b":
                # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
                tokenizer = AutoTokenizer.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.2", cache_dir=cache_dir,
                    padding_side=padding_side, truncation_side=truncation_side)
            case _:
                raise ValueError(f"[DataLoader.get_splits] ValueError: ds_name = {model_name}")

        return tokenizer
