from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class Ranker(ABC):
    """
    In our framework, the Ranker class is responsible for the defining the method of score corrections to be used
    during retrieval. Currently supported methods of score corrections are our proposed method of Nearest Neighbor 
    Normalization, Distribution Normalization (proposed in https://arxiv.org/abs/2302.11084), and a base no-correction
    method. 
    
    A ranker object is initialized with a retriever object, and orchestrates the actual search process by computing
    the necessary score corrections, then using the retriever object to perform search with these score corrections. See
    https://github.com/multimodal-interpretability/nnn for examples of use. 
    """

    @abstractmethod
    def search(self, batch_query: np.matrix, top_k: int) -> Tuple[np.matrix, np.matrix]:
        """
        Searches for the top_k most similar items to the given batch of queries.

        Args:
            batch_query (np.matrix): A batch of query embeddings. Dimensions are (n_queries, embedding_dim).
            top_k (int): The number of top similar items to retrieve.

        Returns:
            The method should return an array of indices of the `top_k` most similar items in order
            for each query in the batch, with dimensions (n_queries, top_k).
        """

        pass
