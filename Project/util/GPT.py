# -*- coding:utf-8 -*-

import os
import json
import logging
import requests

import openai
from openai import OpenAI


class OpenAIGPT:

    def __init__(self):
        env_var = dict(os.environ)
        openai_api_key = env_var["OPENAI_API_KEY"] if "OPENAI_API_KEY" in env_var else ""
        openai_org = env_var["OPENAI_ORG"] if "OPENAI_ORG" in env_var else ""
        openai.api_key = openai_api_key
        openai.organization = openai_org  # Organization name: Personal

        client = OpenAI()
        OpenAI.api_key = openai_api_key

        save_dir = f"./output/"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

#         prompt_template = """You are trying to use the Google Images search engine to
# collect images for a task named "{prompt}".
# \nGive a list of {num_query} search queries for collecting such images.
# \nPlease be accurate and specific about the search engine queries.
# \nYou must output the results as a valid JSON object, which contains only one attribute: "query".
# Data format: the content of the attribute "query" is a list of strings."""
#         self.prompt_template = prompt_template.replace("\n", "").strip()

        self.env_var = env_var
        self.openai_api_key = openai_api_key
        self.client = client
        self.save_dir = save_dir

        logging.basicConfig(
            format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def call_openai_gpt(
            prompt: str,
            max_tokens: int = 150,
            model_name: str = "gpt-3.5-turbo",  # "gpt-4" "gpt-4-1106-preview" "text-davinci-002"
    ) -> str:
        """
        Calls the ChatGPT API with the given prompt.

        Parameters:
        - prompt (str): The prompt to send to the API.
        - max_tokens (int, optional): The maximum length of the response. Defaults to 150.
        - model_name (str, optional): The model to use. Defaults to "gpt-3.5-turbo".

        Returns:
        - str: The generated text.
        """

        response = openai.Completion.create(
            engine=model_name,
            # model=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            # temperature=0,
        )
        return response.choices[0].text.strip()

    def get_full_response(self, prompt: str, max_tokens_per_call: int = 150, total_tokens: int = 500):
        """
        Continuously calls the API until the desired number of tokens is reached.

        Parameters:
        - prompt (str): The initial prompt to send to the API.
        - max_tokens_per_call (int, optional): The maximum tokens for each API call. Defaults to 150.
        - total_tokens (int, optional): The total tokens desired for the full response. Defaults to 500.

        Returns:
        - str: The full generated text.
        """

        remaining_tokens = total_tokens
        current_prompt = prompt

        full_response = self.call_openai_gpt(current_prompt, min(remaining_tokens, max_tokens_per_call))

        return full_response

    def call_gpt_request(
            self,
            prompt: str,
            model_name: str = "gpt-3.5-turbo",  # "gpt-4" "gpt-4-1106-preview" "text-davinci-002"
    ) -> str:
        API_KEY = self.openai_api_key
        API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

        def generate_chat_completion(
                cur_messages,
                cur_model=model_name,
                temperature=0.7,
                # temperature=1,
                max_tokens=None,
        ):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            }

            data = {
                "model": cur_model,
                "messages": cur_messages,
                "temperature": temperature,
            }
            # response_format={"type": "json_object"}

            if max_tokens is not None:
                data["max_tokens"] = max_tokens

            response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        response_text = generate_chat_completion(messages)
        return response_text

    def call_gpt_client(
            self,
            prompt: str,
            model_name: str = "gpt-3.5-turbo",  # "gpt-4" "gpt-4-1106-preview" "text-davinci-002"
            get_json: bool = False,
            save_fn: str = "",  # if save_fn is not "", then save the response into the filepath
    ) -> int:
        try:
            response_format = {"type": "json_object"} if get_json else None
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            cur_res = completion.choices[0].message.content.strip()
            # cur_res = self.call_gpt_request(prompt, model_name=model_name)
            # self.logger.info(f">>> cur_res >>> {cur_res}")
        except Exception as e:  # OpenAI request exception
            self.logger.info(e)
            return 1

        try:
            try_json = json.loads(cur_res.strip())  # json object loading checking
            # Data format checking: assert ...
        except Exception as e:  # the output is not a valid json object
            self.logger.info(e)
            return 1

        if isinstance(save_fn, str) and len(save_fn) > 0:
            save_fp = os.path.join(self.save_dir, f"{save_fn}.json")
            with open(save_fp, "w", encoding="utf-8") as fp_w:
                # cur_jsonl = json.dumps(try_json)
                # fp_w.write(cur_jsonl + "\n")
                json.dump(try_json, fp_w)

        return 0
