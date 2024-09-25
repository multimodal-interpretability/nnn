from .retriever import Retriever
import torch
from tqdm import tqdm
import numpy as np
import faiss


class FaissGPURetriever(Retriever):
    def __init__(self, 
                 embeds_size: int,
                 gpu_device: int,
                 reference_index = None,
                 reference_nprobes = 32,
                 retrieval_index = None,
                 retrieval_nprobes = 32
        ):
        resources = faiss.StandardGpuResources() 
        co = faiss.GpuClonerOptions()
        if reference_index is None:
            # set default reference index to flatip
            cpu_reference_index = faiss.IndexFlatIP(embeds_size) 
        else:
            if reference_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise Exception("FAISS retrieval index must use inner product metric!")
            if retrieval_index.d != embeds_size + 1:
                raise Exception(f"Reference index must have embedding size {embeds_size}!")
            cpu_reference_index = reference_index

        self.reference_index = faiss.index_cpu_to_gpu(resources, gpu_device, cpu_reference_index, co)
        self.reference_index.nprobe = reference_nprobes

        if retrieval_index is None:
            self.retrieval_index = faiss.IndexFlatIP(embeds_size + 1)
        else:
            if retrieval_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise Exception("FAISS retrieval index must use inner product metric!")
            if retrieval_index.d != embeds_size + 1:
                raise Exception(f"Retrieval index must have embedding size {embeds_size + 1} due to added bias dimension!")
            cpu_retrieval_index = retrieval_index

        self.retrieval_index = faiss.index_cpu_to_gpu(resources, gpu_device, cpu_retrieval_index, co)
        self.retrieval_index.nprobe = retrieval_nprobes
        
    def setup_retriever(self, retrieval_embeds, reference_embeds, alternate_ks, batch_size):
        # train your indices here
        self.reference_index.train(reference_embeds)
        alignment_means = self.compute_alignment_means(retrieval_embeds, reference_embeds, alternate_ks, batch_size)
        modified_retrieval_embeds = np.concatenate([retrieval_embeds, alignment_means], axis=1)
        self.faiss_retrieval_index.train(modified_retrieval_embeds)
        
        return alignment_means

    def compute_alignment_means(self, retrieval_embeds, reference_embeds, alternate_ks, batch_size):
        batch_reference_scores, indices = self.reference_index.search(retrieval_embeds, alternate_ks)
        return np.mean(batch_reference_scores, axis=-1, keepdims=True)

    def retrieve(self, retrieval_embeds, batch_query, top_k, alternate_weight, alignment_means, batch_size):
        # append -alt_weight to each vector in the query to account for the -alt_weight * reference_score term
        batch_query = np.concatenate([batch_query, -alternate_weight * np.ones((batch_query.shape[0], 1))], axis=1)
        distances, indices = self.retrieval_index.search(batch_query, top_k)
        return distances, indices