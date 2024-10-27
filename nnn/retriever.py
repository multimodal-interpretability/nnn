from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class Retriever(ABC):
    @abstractmethod
    def compute_alignment_means(
        self, retrieval_embeds, reference_embeds, alternate_ks, batch_size
    ) -> np.matrix:
        pass

    @abstractmethod
    def setup_retriever(
        self, retrieval_embeds, reference_embeds, alternate_ks, batch_size
    ) -> np.matrix:
        pass

    @abstractmethod
    def retrieve(
        self,
        retrieval_embeds,
        batch_query,
        top_k,
        alternate_weight,
        alignment_means,
        batch_size,
    ) -> Tuple[np.matrix, np.matrix]:
        pass
