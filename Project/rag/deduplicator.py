"""
* Author: Juntai Cao
* Date: 2024-04-13
"""

from typing import List, Optional

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


class Deduplicator:

    def __init__(
            self,
            model_name: str = "sentence-transformers/sentence-t5-large",
            device: Optional[str] = None,
            cache_folder: Optional[str] = None,
            trust_remote_code: bool = False,
            revision: Optional[str] = None,
            threshold: float = 0.8
    ):
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        self.threshold = threshold

    def cosine_similarity(self, sentence1: str, sentence2: str) -> float:
        embedding1 = self.model.encode(sentence1)
        embedding2 = self.model.encode(sentence2)
        return 1 - cosine(embedding1, embedding2)

    def deduplicate(self, retrieved: List[str]) -> List[str]:
        base_text = retrieved[0]
        deduplicated_retrieved = [base_text]
        for idx, entry in enumerate(retrieved):
            for selected in deduplicated_retrieved:
                if self.cosine_similarity(entry, selected) < self.threshold:
                    deduplicated_retrieved.append(entry)
        return deduplicated_retrieved
