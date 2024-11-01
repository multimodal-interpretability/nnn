from .retriever import Retriever
from .ranker import Ranker
import torch
import numpy as np


class DNRanker(Ranker):
    def __init__(
        self,
        retriever: Retriever,
        retrieval_embeds: np.matrix,
        reference_embeds: np.matrix,
        alternate_ks: int = 256,
        batch_size: int = 128,
        alternate_weight=0.5,  # can remove this?
        # gpu params
        use_gpu: bool = False,
        gpu_id: int = -1,
        # DN params
        retrieval_dev_embeds=None,
        query_dev_embeds=None,
        dn_lambda=0.5,
    ):
        self.retriever = retriever
        self.alternate_weight = alternate_weight
        self.batch_size = batch_size
        self.alternate_ks = alternate_ks
        self.dn_lambda = dn_lambda

        if use_gpu and gpu_id == -1:
            raise Exception("GPU flag set but no GPU device given!")

        if use_gpu != retriever.use_gpu or gpu_id != retriever.gpu_id:
            raise Exception("Ranker and retriever must use same device!")

        self.device = "cpu" if not use_gpu else f"cuda:{gpu_id}"
        self.embed_size = retrieval_embeds.shape[1]

        if reference_embeds.shape[1] != retrieval_embeds.shape[1]:
            raise Exception(
                "Mismatch in embedding dimensions between retrieval and reference set!"
            )

        if retrieval_dev_embeds is None or query_dev_embeds is None:
            raise Exception(
                "must provide both query and reference dev embeds for distribution normalization!"
            )

        self.retrieval_dev_embeds_avg = np.mean(
            retrieval_dev_embeds, axis=0, keepdims=True
        )
        self.query_dev_embeds_avg = np.mean(query_dev_embeds, axis=0, keepdims=True)

        retrieval_embeds = retrieval_embeds - dn_lambda * self.retrieval_dev_embeds_avg
        reference_embeds = reference_embeds - dn_lambda * self.query_dev_embeds_avg

        self.reference_embeds = reference_embeds
        self.torch_reference_embeds = torch.tensor(reference_embeds, device=self.device)

        self.retrieval_embeds = retrieval_embeds
        self.torch_retrieval_embeds = torch.tensor(retrieval_embeds, device=self.device)

        self.alignment_means = torch.tensor(
            retriever.setup_retriever(
                self.torch_retrieval_embeds,
                self.torch_reference_embeds,
                self.alternate_ks,
                self.batch_size,
            ),
            device=self.device,
        )

    def search(self, batch_query: np.matrix, top_k):
        """
        Searches for the top_k most similar items to the given batch of queries, using the 
        test-time distribution normalization method to modify the similarity scores. 

        Args:
            batch_query (np.matrix): A batch of query embeddings. Dimensions are (n_queries, embedding_dim).
            top_k (int): The number of top similar items to retrieve.

        Returns:
            The method should return an array of indices of the `top_k` most similar items in order
            for each query in the batch, with dimensions (n_queries, top_k).
        """
        torch_batch_query = torch.tensor(
            batch_query - self.dn_lambda * self.query_dev_embeds_avg, device=self.device
        )
        return self.retriever.retrieve(
            self.torch_retrieval_embeds,
            torch_batch_query,
            top_k,
            self.alternate_weight,
            self.alignment_means,
            self.batch_size,
        )
