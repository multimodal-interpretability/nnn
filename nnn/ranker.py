from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import numpy as np

class Ranker(ABC):
    @abstractmethod
    def search(self, batch_query: np.matrix, top_k: int):
        pass