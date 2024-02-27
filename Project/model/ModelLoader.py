# -*- coding:utf-8 -*-

import os
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM


class ModelLoader:

    def __init__(self, logger: logging.Logger = None):
        if isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            logging.basicConfig(
                format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
            )
            self.logger = logging.getLogger(self.__class__.__name__)

    def load_model(
            self,
            model_name: str,
            cache_dir: Optional[str] = None,  # "~/.cache/huggingface/"
            verbose: bool = False,
    ) -> dict:
        """
        Get the model via Hugging Face API and/or init with the local checkpoint.
        :param model_name: model name (for Hugging Face API). https://huggingface.co/models
        :param cache_dir: The directory where data & model are cached.
        :param verbose: Verbose model: show logs.
        :return: the model dict.
        """

        model = self.get_model(model_name=model_name, cache_dir=cache_dir)
        # model.to(DEVICE)
        # model.train()
        # model.eval()

        # Show model information
        if verbose:
            self.logger.info(f"[Model] Parameters (total): {model.num_parameters()}")
            self.logger.info(f"[Model] Parameters (trainable): "
                             f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        return {
            "model": model,
        }

    def get_model(
            self,
            model_name: str = "",
            local_dir: str = "",
            local_path: str = "",
            cache_dir: Optional[str] = None,  # "~/.cache/huggingface/"
    ):
        """
        Get the model via Hugging Face API and/or init with the local checkpoint.
        :param model_name: model name (for Hugging Face API). https://huggingface.co/models
        :param local_dir: the directory of the local checkpoint (from model.save_pretrained(local_dir)).
        :param local_path: file path of the local checkpoint (from torch.save(model.state_dict(), local_path)).
        :param cache_dir: The directory where data & model are cached.
        :return: the model (Causal LM). https://huggingface.co/docs/transformers/en/tasks/language_modeling
        """

        if isinstance(local_dir, str) and os.path.isdir(local_dir):
            # Try to load the local model first
            try:
                model = AutoModelForCausalLM.from_pretrained(local_dir)
                return model
            except Exception as e:
                self.logger.info(f"[get_model] >>> local_dir not effective:\n{e}")

        if not isinstance(model_name, str) or model_name == "":
            raise ValueError(f"[{self.__class__.__name__}] ValueError: model_name = {model_name}")

        model_name = model_name.strip()
        self.logger.info(f"[get_model] >>> Loading model from Hugging Face. model_name: {model_name}")

        if model_name == "gpt1":
            # openai-gpt ("GPT-1") is the first transformer-based language model created and released by OpenAI
            model = AutoModelForCausalLM.from_pretrained("openai-community/openai-gpt", cache_dir=cache_dir)
        elif model_name == "gpt2":
            # The smallest version of GPT-2, with 124M parameters.
            model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", cache_dir=cache_dir)
        elif model_name == "gpt2-medium":
            # GPT-2 Medium is the 355M parameter version of GPT-2
            model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium", cache_dir=cache_dir)
        elif model_name == "gpt2-large":
            # GPT-2 Large is the 774M parameter version of GPT-2
            model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large", cache_dir=cache_dir)
        elif model_name == "gpt2-xl":
            # GPT-2 XL is the 1.5B parameter version of GPT-2
            model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl", cache_dir=cache_dir)
        else:
            raise ValueError(f"[DataLoader.get_splits] ValueError: ds_name = {model_name}")

        if isinstance(local_path, str) and os.path.isfile(local_path):
            try:
                self.logger.info(f"Loading local model checkpoint (state_dict) from: {local_path}")
                model.load_state_dict(torch.load(local_path))
            except Exception as e:
                self.logger.info(f"[get_model] >>> local_path not effective:\n{e}")

        return model
