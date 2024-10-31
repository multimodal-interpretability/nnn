from .retriever import Retriever
import torch
from tqdm import tqdm
import numpy as np


class BaseRetriever(Retriever):
    def __init__(self, embeds_size: int, use_gpu: bool = False, gpu_id: int = -1):
        """
        Initializes a new instance of BaseRetriever.

        Args:
            embeds_size (int): The dimension size of embeddings.
            use_gpu (bool, optional): If True, will attempt to use GPU. Default is False.
            gpu_id (int, optional): Specifies GPU device ID if `use_gpu` is True. 
                                    Default is -1 (no GPU device).

        Raises:
            Exception: If `use_gpu` is True but `gpu_id` is not specified.
        """

        self.embeds_size = embeds_size
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        if self.use_gpu and self.gpu_id == -1:
            raise Exception("GPU flag set but no GPU device given!")

    def compute_alignment_means(
        self, retrieval_embeds, reference_embeds, alternate_ks, batch_size
    ):
        return (torch.zeros((retrieval_embeds.shape[0],), device=retrieval_embeds.device)).cpu().numpy()

    def setup_retriever(
        self, retrieval_embeds, reference_embeds, alternate_ks, batch_size
    ):
        alignment_means = self.compute_alignment_means(
            retrieval_embeds, reference_embeds, alternate_ks, batch_size
        )
        return alignment_means

    def retrieve(
        self,
        retrieval_embeds,
        batch_query,
        top_k,
        alternate_weight,
        alignment_means,
        batch_size,
    ):
        distances = []
        indices = []
        for i in tqdm(range(0, batch_query.shape[0], batch_size)):
            batch_similarity_scores = torch.einsum(
                "ik,jk->ij", batch_query[i : i + batch_size, :], retrieval_embeds
            )
            top_k_results = torch.topk(batch_similarity_scores, top_k, dim=1)
            distances.append(top_k_results.values)
            indices.append(top_k_results.indices)
        return (
            torch.vstack(distances).cpu().numpy(),
            torch.vstack(indices).cpu().numpy(),
        )
