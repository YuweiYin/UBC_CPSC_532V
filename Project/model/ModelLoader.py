# -*- coding:utf-8 -*-
import os.path

import torch
from transformers import AutoModelForCausalLM


class ModelLoader:

    def __init__(self):
        pass

    def get_model(
            self,
            model_name: str = "",
            local_path: str = "",
            cache_dir: str = "~/.cache/huggingface/datasets",
    ):
        """
        Get the model via Hugging Face API and/or init with the local checkpoint.
        :param model_name: model name (for Hugging Face API). https://huggingface.co/models
        :param local_path: file path of the local checkpoint.
        :param cache_dir: The directory where data & model are cached.
        :return: the model (Causal LM). https://huggingface.co/docs/transformers/en/tasks/language_modeling
        """

        if not isinstance(model_name, str) or model_name == "":
            raise ValueError(f"[{self.__class__.__name__}] ValueError: model_name = {model_name}")

        model_name = model_name.strip()
        print(f"Loading model from Hugging Face. model_name: {model_name}")

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
                print(f"Loading local model checkpoint from: {local_path}")  # TODO: test loading local model ckpt
                model.load_state_dict(torch.load(local_path))
            except Exception as e:
                print(e)

        return model
