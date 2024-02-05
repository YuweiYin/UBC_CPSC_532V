# -*- coding:utf-8 -*-

import requests
from typing import Tuple


class ConceptNet:

    def __init__(self):
        self.prefix_url = "https://api.conceptnet.io"
        self.prefix_cid = "/c/en/"  # the prefix of every concept_id

    def get_concept_id(self, word: str) -> str:
        return self.prefix_cid + word.strip()

    def get_url(self, cid: str) -> str:
        return self.prefix_url + cid.strip()

    @staticmethod
    def get_concept(url: str, verbose: bool = True) -> dict:
        try:
            response = requests.get(url.strip()).json()
        except Exception as e:
            if verbose:
                print(f">>> >>> get_concept Exception: {e}")
            response = dict()
        return response

    # def get_edges(self, url: str) -> list:
    #     response = requests.get(url).json()
    #     return response["edges"]

    @staticmethod
    def get_next_node(cid: str, edge: dict) -> Tuple[str, float, str, int]:
        # next_id = edge["end"]["@id"] if edge["end"]["@id"] != cid else edge["start"]["@id"]
        if edge["end"]["@id"] != cid:
            next_id, direction = edge["end"]["@id"], 1
        else:
            next_id, direction = edge["start"]["@id"], 0
        return next_id, edge["weight"], edge["rel"]["label"], direction
