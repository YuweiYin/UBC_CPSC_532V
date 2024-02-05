# -*- coding:utf-8 -*-

from typing import List
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util


class TextParser:

    def __init__(self):
        self.kw_model = KeyBERT()
        self.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def get_keywords_keybert(self, text: str, n_bag: int = 1) -> List[str]:
        keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, n_bag), stop_words="english")
        keywords = [w[0].strip() for w in keywords]
        return keywords

    def keyword_sort(self, references: List[str], keywords: List[str]) -> List[str]:
        """
        sort keywords by computing the average similarity between the current kw and each ref in references
        """

        # # let the selected keywords in choices (will be src_word) not be the same as the tgt_word
        # ref_set = set(references)
        # kw_set = set(keywords)
        # kw_set = kw_set - ref_set
        # keywords = list(kw_set)

        if len(keywords) <= 1:
            return keywords
        assert len(references) >= 1, f"Assertion error: len(references) is {len(references)}, NOT >= 1"

        ref_emb_list = [self.st_model.encode(w, convert_to_tensor=True) for w in references]
        kw_emb_list = [self.st_model.encode(w, convert_to_tensor=True) for w in keywords]

        kw_simi_list = []
        for idx, kw_emb in enumerate(kw_emb_list):
            kw_word = keywords[idx]
            simi_list = [float(util.pytorch_cos_sim(kw_emb, ref_emb)) for ref_emb in ref_emb_list]
            assert len(simi_list) >= 1
            avg_simi = float(sum(simi_list) / len(simi_list))
            kw_simi_list.append((kw_word, avg_simi))

        kw_simi_list.sort(key=lambda x: x[1], reverse=True)
        res_list = [kw_simi[0] for kw_simi in kw_simi_list]

        # res_list = []
        # most_similar_word = ""  # in case no words are similar enough, res_list = [most_similar_word]
        # best_similarity = 0.0
        #
        # for idx, kw_emb in enumerate(kw_emb_list):
        #     kw_word = keywords[idx]
        #     simi_list = [float(util.pytorch_cos_sim(kw_emb, ref_emb)) for ref_emb in ref_emb_list]
        #     max_simi = max(simi_list)
        #     if max_simi >= similarity_limit:
        #         res_list.append(kw_word)
        #     if max_simi >= best_similarity:
        #         most_similar_word = kw_word
        #
        # if len(res_list) == 0:
        #     res_list.append(most_similar_word)

        return res_list
