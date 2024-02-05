# -*- coding:utf-8 -*-

import os
import json

import openai
from openai import OpenAI

from data.rel_template import REL_TO_TEMPLATE


class TextConverter:

    def __init__(self, path, use_gpt: bool, openai_api_key: str = None):
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        else:
            openai_api_key = "YOUR_KEY"  # TODO: Use your own OPENAI_API_KEY if use_gpt is True
        openai.api_key = openai_api_key
        self.use_gpt = use_gpt

        self.node_list, self.w_list, self.r_list = path
        self.openai_api_key = openai_api_key

    def convert(self, e_id: str, c_id: str, saved_dir: str = "output_text", verbose: bool = True) -> None:
        if self.use_gpt and not self.openai_api_key:
            print(f"Error: openai_api_key is {self.openai_api_key}, will use ConceptNet relation templates instead.")
            self.use_gpt = False

        node_relation_list = []
        prompt_list = []
        description_list = []
        assert len(self.node_list) == len(self.w_list) + 1 == len(self.r_list) + 1

        for i in range(len(self.r_list)):
            if self.r_list[i][1] == 0:  # forward
                src_rel_tgt = self.node_list[i], self.r_list[i][0], self.node_list[i + 1]
            else:  # backward
                src_rel_tgt = self.node_list[i + 1], self.r_list[i][0], self.node_list[i]

            start_word = TextConverter.extract_word(src_rel_tgt[0])
            rel = src_rel_tgt[1]
            end_word = TextConverter.extract_word(src_rel_tgt[2])

            # prompt = TextConverter.generate_question(src_rel_tgt[0], src_rel_tgt[1], src_rel_tgt[2])
            # prompt_list.append(prompt)
            # node_relation_list.append(src_rel_tgt)

            if self.use_gpt:
                prompt = TextConverter.generate_question(start_word, rel, end_word)
                cur_description = self.generate_description(prompt).strip()
            else:
                prompt = ""
                cur_description = REL_TO_TEMPLATE[rel.lower()].replace("[w1]", start_word).replace("[w2]", end_word)

            if verbose:
                print(cur_description)

            node_relation_list.append(src_rel_tgt)
            prompt_list.append(prompt)
            description_list.append(cur_description)

        start_node = TextConverter.extract_word(self.node_list[0])
        end_node = TextConverter.extract_word(self.node_list[-1])

        json_data = {
            "example_id": e_id,
            "choice_id": c_id,
            "start_node": start_node,
            "end_node": end_node,
            "node_list": self.node_list,
            "w_list": self.w_list,
            "r_list": self.r_list,
            "node_relation_list": node_relation_list,
            "prompt_list": prompt_list,
            "description_list": description_list,
        }

        start_node = start_node.replace(" ", "_")
        end_node = end_node.replace(" ", "_")
        file_name = f"{e_id}_{c_id}---{start_node}-{end_node}.json"
        output_path = os.path.join(saved_dir, file_name)

        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        with open(output_path, "w") as outfile:
            json.dump(json_data, outfile)

    def generate_description(self, prompt: str, model_name: str = "gpt-3.5-turbo"):
        client = OpenAI(
            api_key=self.openai_api_key,
        )
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return chat_completion.choices[0].message.content

    @staticmethod
    def extract_word(node: str) -> str:
        node_words = node.split("/")
        concept_word = node_words[3].replace("_", " ").strip()
        return concept_word

    @staticmethod
    def generate_question(start_word: str, rel: str, end_word: str) -> str:
        from data.prompt import PROMPT

        # [Optional] use few-shot prompt, where examples are randomly drew from a pool (by BFS on ConceptNet)
        # prompt = f"Write a short sentence only using the words and relationship: {start_word} {rel} {end_word}."
        prompt = PROMPT.format(start_word=start_word, end_word=end_word, relation=rel)
        return prompt
