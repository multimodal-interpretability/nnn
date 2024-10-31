from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class Retriever(ABC):
    @abstractmethod
    def compute_alignment_means(
        self, retrieval_embeds, reference_embeds, alternate_ks, batch_size
    ) -> np.matrix:
        """
        Not intended to be a public facing method; should be called internally in the Ranker interface. 

        Computes mean similarity scores between per-retrieval embedding, using a given set reference embeddings.

        Args:
            retrieval_embeds (torch.Tensor): Embeddings from the retrieval set.
            reference_embeds (torch.Tensor): Reference embeddings to compare against.
            alternate_ks (int): Number of top-k reference scores to average across.
            batch_size (int): Number of samples to process per batch.

        Returns:
            numpy.ndarray: Array of alignment means per retrieval embedding.
        """
        pass

    @abstractmethod
    def setup_retriever(
        self, retrieval_embeds, reference_embeds, alternate_ks, batch_size
    ) -> np.matrix:
        """
        Not intended to be a public facing method; should be called internally in the Ranker interface. 

        Sets up the retriever by calculating alignment means for retrieval.

        Args:
            retrieval_embeds (torch.Tensor): Embeddings from the retrieval set.
            reference_embeds (torch.Tensor): Reference embeddings to compare against.
            alternate_ks (int): Number of top-k reference scores to average.
            batch_size (int): Number of samples to process per batch.

        Returns:
            numpy.ndarray: Array of alignment means per retrieval embedding.
        """
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
        """
        Not intended to be a public facing method; all retrieval should be done from the Ranker interface. 

        Retrieves the top_k most similar items for a batch of query embeddings.

        Args:
            retrieval_embeds (torch.Tensor): Embeddings from the retrieval set.
            batch_query (torch.Tensor): Query embeddings to retrieve items for.
            top_k (int): Number of top items to retrieve.
            alternate_weight (float): Weight to adjust alignment means in similarity scores.
            alignment_means (torch.Tensor): Precomputed alignment means per retrieval embedding to adjust retrieval scores.
            batch_size (int): Number of samples to process per batch.

        Returns:
            tuple: Contains two numpy arrays:
                - distances: The top-k similarity scores for each query.
                - indices: The indices of the top-k retrieved items.
        """
        pass
