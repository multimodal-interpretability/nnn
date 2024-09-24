import torch
import os
import pickle
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import faiss


class NNNRetriever:
    def __init__(
        self, 
        retrieval_embeds: np.matrix,
        reference_embeds: np.matrix,
        alternate_ks: int = 256,
        batch_size: int = 128,
        alt_weight = 0.5,
        # GPU-related params
        use_gpu: bool = False,
        gpu_id: int = -1,
        # FAISS params
        use_faiss_reference: bool = False,
        faiss_reference_index = None,
        use_faiss_retrieval: bool = False,
        faiss_retrieval_index = None,
        # Distribution Normalization params
        distribution_normalization = False,
        retrieval_dev_embeds = None,
        query_dev_embeds = None,
        dn_lambda = 0.5
        ):

        self.alt_weight = alt_weight
        self.batch_size = batch_size
        self.alternate_ks = alternate_ks
        
        if use_gpu and gpu_id == -1:
            raise Exception("GPU flag set but no GPU device given!")
        
        self.device = 'cpu' if not use_gpu else f'cuda:{gpu_id}'
        self.embed_size = retrieval_embeds.shape[1]
        self.distribution_normalization = distribution_normalization
        self.dn_lambda = dn_lambda

        if reference_embeds.shape[1] != retrieval_embeds.shape[1]:
            raise Exception("Mismatch in embedding dimensions between retrieval and reference set!")
        
        if distribution_normalization:
            if retrieval_dev_embeds is None or query_dev_embeds is None:
                raise Exception("must provide both query and reference dev embeds for distribution normalization!")
        
            self.retrieval_dev_embeds_avg = np.mean(retrieval_dev_embeds, axis=0, keepdims=True)
            self.query_dev_embeds_avg = np.mean(query_dev_embeds, axis=0, keepdims=True)

            retrieval_embeds = retrieval_embeds - dn_lambda * self.retrieval_dev_embeds_avg
            reference_embeds = reference_embeds - dn_lambda * self.query_dev_embeds_avg

        self.reference_embeds = reference_embeds
        self.torch_reference_embeds = torch.tensor(reference_embeds, device=self.device)
        self.use_faiss_reference = use_faiss_reference

        self.retrieval_embeds = retrieval_embeds
        self.torch_retrieval_embeds = torch.tensor(retrieval_embeds, device=self.device)
        self.use_faiss_retrieval = use_faiss_retrieval

        if use_faiss_reference:
            self.setup_reference_faiss_index(reference_embeds, faiss_reference_index, use_gpu, gpu_id)
        # compute the alternate scores from the reference set here
        reference_mean_scores = self.compute_alignment_means(alternate_ks)
        self.reference_mean_scores = reference_mean_scores
        if use_faiss_reference and not use_faiss_retrieval:
            self.reference_mean_scores = torch.tensor(reference_mean_scores, device=self.device)
        if not use_faiss_reference and use_faiss_retrieval:
            self.reference_mean_scores = reference_mean_scores.cpu().numpy()
        
        if use_faiss_retrieval:
            self.setup_retrieval_faiss_index(retrieval_embeds, faiss_retrieval_index, use_gpu, gpu_id)

    def setup_reference_faiss_index(self, reference_embeds, faiss_reference_index, use_gpu, gpu_id):
        if faiss_reference_index != None:
            if faiss_reference_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise Exception("FAISS reference index must use inner product metric!")
            elif faiss_reference_index.d != self.embed_size:
                raise Exception(f"FAISS reference index must be of size {self.embed_size}")
            else:
                self.faiss_reference_index = faiss_reference_index
        else:
            base_nlist = 2048
            quantizer = faiss.IndexFlatIP(self.embed_size)
            cpu_index = faiss.IndexIVFFlat(quantizer, self.embed_size, nlist, faiss.METRIC_INNER_PRODUCT)
            if use_gpu:
                res = faiss.StandardGpuResources() 
                co = faiss.GpuClonerOptions()
                gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index, co)
                self.faiss_reference_index = gpu_index
            else:
                self.faiss_reference_index = cpu_index

        self.faiss_reference_index.train(self.reference_embeds)
        self.faiss_reference_index.add(self.reference_embeds)
        # TODO: get rid of this magic constant
        self.faiss_reference_index.nprobe = 32

    
    def setup_retrieval_faiss_index(self, retrieval_embeds, faiss_retrieval_index, use_gpu, gpu_id):
        if faiss_retrieval_index != None:
            if faiss_retrieval_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise Exception("FAISS retrieval index must use inner product metric!")
            elif faiss_retrieval_index.d != self.embed_size + 1:
                raise Exception(f"FAISS retrieval index must have size {self.embed_size + 1}!")
            else:
                self.faiss_retrieval_index = faiss_retrieval_index
        else:
            # IMPORTANT: we add 1 to the dimension since we append the reference scores to the embeds
            cpu_index = faiss.IndexFlatIP(self.embed_size + 1)
            if use_gpu:
                res = faiss.StandardGpuResources() 
                gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
                self.faiss_retrieval_index = gpu_index
            else:
                self.faiss_retrieval_index = cpu_index
        
        # train and add
        modified_retrieval_embeds = np.concatenate([retrieval_embeds, self.reference_mean_scores], axis=1)
        self.faiss_retrieval_index.train(modified_retrieval_embeds)
        self.faiss_retrieval_index.add(modified_retrieval_embeds)
        # TODO: get rid of this magic constant
        self.faiss_reference_index.nprobe = 32
            
    def compute_alignment_means(self, alternate_ks):
        if self.use_faiss_reference:
            return self.compute_alignment_means_faiss(alternate_ks)
        else:
            return self.compute_alignment_means_exhaustive(alternate_ks)

    def compute_alignment_means_faiss(self, alternate_ks):
        batch_reference_scores, indices = self.faiss_reference_index.search(self.retrieval_embeds, alternate_ks)
        return np.mean(batch_reference_scores, axis=-1, keepdims=True)

    def compute_alignment_means_exhaustive(self, alternate_ks):
        alignment_means = []
        for i in tqdm(range(0, self.torch_retrieval_embeds.shape[0], self.batch_size)):
            batch_reference_similarity_scores = torch.einsum("ik,jk->ij", self.torch_retrieval_embeds[i:i+self.batch_size, :], self.torch_reference_embeds)
            top_k_reference_scores = torch.topk(batch_reference_similarity_scores, alternate_ks, dim=1)
            alignment_means.append(torch.mean(top_k_reference_scores.values, dim=1, keepdim=True))
        return torch.cat(alignment_means)

    def search(self, batch_query: np.matrix, top_k):
        if self.distribution_normalization:
            batch_query = batch_query - self.dn_lambda * self.query_dev_embeds_avg
        if self.use_faiss_retrieval:
            return self.search_faiss(batch_query, top_k)
        else:
            return self.search_exhaustive(torch.tensor(batch_query, device=self.device), top_k)
        
    def search_faiss(self, batch_query: np.matrix, top_k, alt_weight = None):
        alt_weight_to_use = self.alt_weight
        if alt_weight:
            alt_weight_to_use = alt_weight

        # append -alt_weight to each vector in the query to account for the -alt_weight * reference_score term
        batch_query = np.concatenate([batch_query, -alt_weight_to_use * np.ones((batch_query.shape[0], 1))], axis=1)
        distances, indices = self.faiss_retrieval_index.search(batch_query, top_k)
        return distances, indices

    def search_exhaustive(self, batch_query: torch.tensor, top_k, alt_weight = None):
        alt_weight_to_use = self.alt_weight
        if alt_weight:
            alt_weight_to_use = alt_weight

        distances = []
        indices = []
        print("DIMENSIONS ARE", batch_query.shape, self.torch_retrieval_embeds.shape, self.reference_mean_scores.shape)
        for i in tqdm(range(0, batch_query.shape[0], self.batch_size)):
            batch_similarity_scores = torch.einsum("ik,jk->ij", batch_query[i:i+self.batch_size, :], self.torch_retrieval_embeds) - alt_weight_to_use * self.reference_mean_scores.T
            top_k_results = torch.topk(batch_similarity_scores, top_k, dim=1)
            distances.append(top_k_results.values)
            indices.append(top_k_results.indices)
        return torch.vstack(distances).cpu().numpy(), torch.vstack(indices).cpu().numpy()
