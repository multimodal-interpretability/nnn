from abc import ABC, abstractmethod
import numpy as np


class Ranker(ABC):
    @abstractmethod
    def search(self, batch_query: np.matrix, top_k: int):
        """
        Abstract method to search for the top_k most similar items to the given batch of queries.

        Args:
            batch_query (np.matrix): A batch of query embeddings. Dimensions are (n_queries, embedding_dim).
            top_k (int): The number of top similar items to retrieve.

        Returns:
            The method should return an array of indices of the `top_k` most similar items in order
            for each query in the batch, with dimensions (n_queries, top_k).
        """

        pass
