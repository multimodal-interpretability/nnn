from .retriever import Retriever
import torch
from tqdm import tqdm
import numpy as np


class BaseRetriever(Retriever):
    def __init__(self, 
                 embeds_size: int, 
                 use_gpu: bool = False, 
                 gpu_id: int = -1
        ):

        self.embeds_size = embeds_size
        self.use_gpu = use_gpu
        self.gpu_id = -1

        if self.use_gpu and self.gpu_id == -1:
            raise Exception("GPU flag set but no GPU device given!")
        
    def compute_alignment_means(self, retrieval_embeds, reference_embeds, alternate_ks, batch_size):
        alignment_means = []
        for i in tqdm(range(0, retrieval_embeds.shape[0], batch_size)):
            batch_reference_similarity_scores = torch.einsum(retrieval_embeds[i : i + batch_size, :], reference_embeds)
            top_k_reference_scores = torch.topk(batch_reference_similarity_scores, alternate_ks, dim=1)
            alignment_means.append(torch.mean(top_k_reference_scores.values, dim=1, keepdim=True))
        return torch.cat(alignment_means)
    
    def setup_retriever(self, retrieval_embeds, reference_embeds, alternate_ks, batch_size):
        alignment_means = self.compute_alignment_means(retrieval_embeds, reference_embeds, alternate_ks, batch_size)
        return alignment_means
    
    def retrieve(self, retrieval_embeds, batch_query, top_k, alternate_weight, alignment_means, batch_size):
        distances = []
        indices = []
        for i in tqdm(range(0, batch_query.shape[0], batch_size)):
            batch_similarity_scores = torch.einsum("ik,jk->ij", batch_query[i:i + batch_size, :], retrieval_embeds) - alternate_weight * alignment_means.T
            top_k_results = torch.topk(batch_similarity_scores, top_k, dim=1)
            distances.append(top_k_results.values)
            indices.append(top_k_results.indices)
        return torch.vstack(distances).cpu().numpy(), torch.vstack(indices).cpu().numpy()