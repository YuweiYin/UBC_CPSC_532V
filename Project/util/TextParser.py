# -*- coding:utf-8 -*-

from typing import List
from keybert import KeyBERT


class TextParser:

    def __init__(self):
        self.kw_model = KeyBERT()

    def get_keywords_keybert(self, text: str, n_bag: int = 1) -> List[str]:
        keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, n_bag), stop_words="english")
        keywords = [w[0].strip() for w in keywords]
        return keywords
