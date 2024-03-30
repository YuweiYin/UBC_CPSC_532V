import os
import json
import requests
import wikipediaapi
import arxiv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

from .openai_setup import OPENAI_API_KEY
from .conceptnet_setup import PREFIX_URL, REL_TO_TEMPLATE
from .prompt import PROMPT


class Retriever:

    def __init__(self):
        self.name = self.__class__.__name__

    def retrieve(self, query: str, **kwargs):
        raise NotImplementedError("The retrieve method must be implemented in subclasses.")


class AtomicRetriever(Retriever):

    def __init__(self):
        super().__init__()

    def retrieve(self, query: str, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("mismayil/comet-gpt2-ai2")
        model = AutoModelForCausalLM.from_pretrained("mismayil/comet-gpt2-ai2").to(device)

        # query = "I enjoy walking with my cute dog"
        inputs = tokenizer(query, return_tensors="pt").to(device)

        generated_ids = model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
        retrieved = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return retrieved


class GPTRetriever(Retriever):
    """
    Note: please set a valid OPENAI_API_KEY in openai_setup.py
    """

    def __init__(self, api_key: str = OPENAI_API_KEY, model_name: str = "gpt-3.5-turbo", verbose: bool = False):
        super().__init__()
        self.api_key = api_key
        self.model = model_name
        self.verbose = verbose

    def retrieve(self, query: str, **kwargs):
        try:
            client = OpenAI(
                api_key=self.api_key,
            )
            chat_completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": PROMPT.format(query=query)}
                ],
            )
            retrieved = chat_completion.choices[0].message.content
        except Exception as e:
            if self.verbose:
                print(f">>> >>> GPTRetriever - retrieve Exception: {e}")
            retrieved = None

        return retrieved


class WikiRetriever(Retriever):
    """
    Note: the query of the WiKi retriever should be words or phrases.
    """

    def __init__(self, full_text: bool = False):
        super().__init__()
        self.full_text = full_text
        self.wiki = wikipediaapi.Wikipedia("Wiki_retrieval_agent", "en", extract_format=wikipediaapi.ExtractFormat.WIKI)

    def retrieve(self, query: str, **kwargs):
        if not self.wiki.page(query).exists():
            pass
        else:
            wiki_page = self.wiki.page(query)
            retrieved = wiki_page.text if self.full_text else wiki_page.summary

            if os.path.exists("Wiki_retrieved.json"):
                with open("Wiki_retrieved.json", "r") as file:
                    data = json.load(file)
            else:
                data = []

            data.append({"question": query, "retrieved_text": retrieved})

            with open("Wiki_retrieved.json", "w") as file:
                json.dump(data, file, indent=4)

            return retrieved


class ConceptNetRetriever(Retriever):
    """
    Note: the query of the ConceptNet retriever should be words or phrases.
    """

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose

    def get_concept(self, query: str):
        url = PREFIX_URL + query.strip()
        try:
            response = requests.get(url.strip()).json()
        except Exception as e:
            if self.verbose:
                print(f">>> >>> get_concept Exception: {e}")
            response = dict()
        return response

    def retrieve(self, query: str, **kwargs):
        edges = self.get_concept(query)["edges"]
        descriptions = []
        for edge in edges:
            if edge["start"]["language"] == "en" and edge["end"]["language"] == "en":
                start_word = edge["start"]["label"]
                end_word = edge["end"]["label"]
                rel = edge["rel"]["label"]
                descriptions.append(REL_TO_TEMPLATE[rel.lower()].replace("[w1]", start_word).replace("[w2]", end_word))
        retrieved = ". ".join(descriptions)

        return retrieved


class ArxivRetriever(Retriever):

    def __init__(self):
        super().__init__()
        self.client = arxiv.Client()

    def retrieve(self, query: str, max_results: int = 10, **kwargs):
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        retrieved = ""
        for r in self.client.results(search):
            retrieved = retrieved + r.summary + "\n\n"

        return retrieved
