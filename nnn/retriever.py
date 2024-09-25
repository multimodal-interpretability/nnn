from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import numpy as np

class Retriever(ABC):
    @abstractmethod
    def compute_alignment_means(self, retrieval_embeds, reference_embeds, alternate_ks, batch_size):
        pass

    @abstractmethod
    def setup_retriever(self, retrieval_embeds, reference_embeds, alternate_ks, batch_size):
        pass

    @abstractmethod
    def retrieve(self, retrieval_embeds, batch_query, top_k, alternate_weight, alignment_means, batch_size):
        pass