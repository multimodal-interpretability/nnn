from .retriever import Retriever
import torch
import numpy as np
import faiss

class FaissCPURetriever(Retriever):
    def __init__(
        self,
        embeds_size: int,
        reference_index=None,
        reference_nprobes=32,
        retrieval_index=None,
        retrieval_nprobes=32,
    ):
        """
        Initializes a new instance of FaissCPURetriever.

        Args:
            embeds_size (int): The dimension size of embeddings.
            reference_index (Faiss index): a Faiss index to be used to search reference embeddings. Should use inner product metric, as this is the distance metric we are operating in.
            reference_nprobes (int): See nprobes for Faiss indices.
            retrieval_index (Faiss index): a Faiss index to be used to search reference embeddings. Should use inner product metric, as this is the distance metric we are operating in.
            retrieval_nprobes (int): See nprobes for Faiss indices.
        """
        self.gpu_id = -1
        self.use_gpu = False

        if reference_index is None:
            # set default reference index to flatip
            self.reference_index = faiss.IndexFlatIP(embeds_size)
            self.reference_index.nprobe = reference_nprobes
        else:
            if reference_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise Exception("FAISS retrieval index must use inner product metric!")
            self.reference_index = reference_index
            self.reference_index.nprobe = reference_nprobes

        if retrieval_index is None:
            self.retrieval_index = faiss.IndexFlatIP(embeds_size + 1)
            self.retrieval_index.nprobe = retrieval_nprobes
        else:
            if retrieval_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise Exception("FAISS retrieval index must use inner product metric!")
            if retrieval_index.d != embeds_size + 1:
                raise Exception(
                    f"Retrieval index must have embedding size {embeds_size + 1} due to added bias dimension!"
                )

    def setup_retriever(
        self, retrieval_embeds, reference_embeds, alternate_ks, batch_size
    ):
        # train your indices here
        # check that the dimensions are the right sizes
        self.check_dimensions(retrieval_embeds)
        if not self.reference_index.is_trained:
            self.reference_index.train(reference_embeds.numpy())
        self.reference_index.add(reference_embeds.numpy())
        alignment_means = self.compute_alignment_means(
            retrieval_embeds, reference_embeds, alternate_ks, batch_size
        )
        modified_retrieval_embeds = np.concatenate(
            [retrieval_embeds.numpy(), alignment_means], axis=1
        )
        if not self.retrieval_index.is_trained:
            self.retrieval_index.train(modified_retrieval_embeds)
        self.retrieval_index.add(modified_retrieval_embeds)

        return alignment_means

    def compute_alignment_means(
        self,
        retrieval_embeds: torch.Tensor,
        reference_embeds: torch.Tensor,
        alternate_ks,
        batch_size,
    ):
        self.check_dimensions(retrieval_embeds)
        batch_reference_scores, indices = self.reference_index.search(
            retrieval_embeds.numpy(), alternate_ks
        )
        return np.mean(batch_reference_scores, axis=-1, keepdims=True)

    def retrieve(
        self,
        retrieval_embeds: torch.Tensor,
        batch_query: torch.Tensor,
        top_k: int,
        alternate_weight: float,
        alignment_means: torch.Tensor,
        batch_size: int,
    ):
        self.check_dimensions(retrieval_embeds)
        # append -alt_weight to each vector in the query to account for the -alt_weight * reference_score term
        batch_query = batch_query.numpy()
        batch_query = np.concatenate(
            [batch_query, -alternate_weight * np.ones((batch_query.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        distances, indices = self.retrieval_index.search(batch_query, top_k)
        return distances, indices

    def check_dimensions(self, retrieval_embeds: torch.Tensor):
        if self.reference_index.d != retrieval_embeds.shape[1]:
            raise Exception(
                "Reference index embedding size does not match with reference embeds!"
            )
        if self.retrieval_index.d != retrieval_embeds.shape[1] + 1:
            raise Exception(
                "Retrieval index embedding size does not match with retrieval embeds!"
            )
