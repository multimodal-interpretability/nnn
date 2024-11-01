from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class Retriever(ABC):
    """
    In our framework, the Retriever class is responsible for the actual retrieval mechanisms. It is intended
    to define the method of retrieval (exhaustive search using Pytorch, narest-neighbor vector search using Faiss), and should
    be used with a Ranker class that defines the actual method of score corrections to be used during retrieval (which includes Nearest 
    Neighbor Normalization). See the docs for the Ranker class for more detail.

    Note that the retriever class does not store any information locally, and depends on passed in values for retrieval embeddings, 
    query embeddings, reference embeddings, and/or additive corrections, which what define the actual Nearest Neighbor Normalization 
    method. Thus, it should be used with a ranker object, rather than directly used. 
    """

    @abstractmethod
    def compute_alignment_means(
        self, retrieval_embeds, reference_embeds, alternate_ks, batch_size
    ) -> np.matrix:
        """
        Not intended to be a public facing method; should be called internally in the Ranker interface. 

        Computes mean similarity scores between per-retrieval embedding, using a given set reference embeddings.
        As described in the paper, these mean scores are later used as an additive correction to the base retrieval scores.

        Args:
            retrieval_embeds (torch.Tensor): Embeddings from the retrieval set.
            reference_embeds (torch.Tensor): Reference embeddings to compare against.
            alternate_ks (int): Number of top-k reference scores to average across.i
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

        Sets up the retriever by calculating alignment means for retrieval, allowing for alignment
        means to be used at inference time for retrieval as described in the paper. 

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
