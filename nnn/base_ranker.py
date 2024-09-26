from .retriever import Retriever
from .ranker import Ranker
import torch
import numpy as np


class BaseRanker(Ranker):
    def __init__(
        self,
        retriever: Retriever,
        retrieval_embeds: np.matrix,
        reference_embeds: np.matrix,
        batch_size: int = 128,
        # gpu params
        use_gpu: bool = False,
        gpu_id: int = -1,
    ):
        self.retriever = retriever
        self.batch_size = batch_size

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

        self.reference_embeds = reference_embeds
        self.torch_reference_embeds = torch.tensor(reference_embeds, device=self.device)

        self.retrieval_embeds = retrieval_embeds
        self.torch_retrieval_embeds = torch.tensor(retrieval_embeds, device=self.device)

        self.alignment_means = torch.zeros(
            (self.torch_retrieval_embeds.shape[0]), device=self.device
        )  # torch.tensor(retriever.setup_retriever(self.torch_retrieval_embeds, self.torch_reference_embeds, self.alternate_ks, self.batch_size), device=self.device)

    def search(self, batch_query: np.matrix, top_k):
        torch_batch_query = torch.tensor(batch_query, device=self.device)
        return self.retriever.retrieve(
            self.torch_retrieval_embeds,
            torch_batch_query,
            top_k,
            0,
            self.alignment_means,
            self.batch_size,
        )
