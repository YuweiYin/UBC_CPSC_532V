import os
import json
import requests
from typing import List
import wikipediaapi
import arxiv
from googlesearch import search as g_search

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("mismayil/comet-gpt2-ai2")
        self.model = AutoModelForCausalLM.from_pretrained("mismayil/comet-gpt2-ai2").to(self.device)

    def retrieve(self, query: str, **kwargs) -> List[str]:
        retrieved = []

        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
        retrieved.append(self.tokenizer.decode(generated_ids[0], skip_special_tokens=True))

        return retrieved


class GPTRetriever(Retriever):
    """
    Note: please set a valid OPENAI_API_KEY in openai_setup.py
    """

    def __init__(self, api_key: str = OPENAI_API_KEY, model_name: str = "gpt-3.5-turbo"):
        super().__init__()
        self.api_key = api_key
        self.model = model_name

    def retrieve(self, query: str, verbose: bool = False, **kwargs) -> List[str]:
        retrieved = []

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
            retrieved.append(chat_completion.choices[0].message.content)
        except Exception as e:
            if verbose:
                print(e)

        return retrieved


class WikiRetriever(Retriever):
    """
    Note: the query of the WiKi retriever should be words or phrases.
    """

    def __init__(self, full_text: bool = False):
        super().__init__()
        self.full_text = full_text
        self.wiki = wikipediaapi.Wikipedia("Wiki_retrieval_agent", "en", extract_format=wikipediaapi.ExtractFormat.WIKI)

    def retrieve(self, query: str, verbose: bool = False, **kwargs) -> List[str]:
        retrieved = []
        try:
            if self.wiki.page(query).exists():
                wiki_page = self.wiki.page(query)
                retrieved.append(wiki_page.text if self.full_text else wiki_page.summary)

                # if os.path.exists("Wiki_retrieved.json"):
                #     with open("Wiki_retrieved.json", "r") as file:
                #         data = json.load(file)
                # else:
                #     data = []

                # data.append({"question": query, "retrieved_text": retrieved})
                # with open("Wiki_retrieved.json", "w") as file:
                #     json.dump(data, file, indent=4)
        except Exception as e:
            if verbose:
                print(e)

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

    def retrieve(self, query: str, verbose: bool = False, **kwargs) -> List[str]:
        retrieved = []

        edges = self.get_concept(query)["edges"]
        for edge in edges:
            try:
                if "language" in edge["start"] and edge["start"]["language"] == "en" and \
                        "language" in edge["end"] and edge["end"]["language"] == "en":
                    start_word = edge["start"]["label"]
                    end_word = edge["end"]["label"]
                    rel = edge["rel"]["label"]
                    cur_desc = REL_TO_TEMPLATE[rel.lower()].replace("[w1]", start_word).replace("[w2]", end_word)
                    retrieved.append(cur_desc)
            except Exception as e:
                if verbose:
                    print(e)
                continue

        return retrieved


class ArxivRetriever(Retriever):

    def __init__(self):
        super().__init__()
        self.client = arxiv.Client()

    def retrieve(self, query: str, max_results: int = 10, verbose: bool = False, **kwargs) -> List[str]:
        retrieved = []

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for r in self.client.results(search):
                retrieved.append(r.summary)
        except Exception as e:
            if verbose:
                print(e)

        return retrieved


class GoogleSearchRetriever(Retriever):

    def __init__(self):
        super().__init__()

    def retrieve(self, query: str, num_results: int = 10, verbose: bool = False, **kwargs) -> List[str]:
        retrieved = []

        try:
            results = g_search(query, num_results=num_results, advanced=True)
            for idx, result in enumerate(results, start=1):
                # retrieved.append(f"{idx}. {result.description}")
                # retrieved.append(f"{result.title}: {result.description}")
                retrieved.append(result.description)
        except Exception as e:
            if verbose:
                print(e)

        return retrieved
