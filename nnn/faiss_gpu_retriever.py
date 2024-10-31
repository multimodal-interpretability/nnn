from .retriever import Retriever
import numpy as np
import faiss


class FaissGPURetriever(Retriever):
    def __init__(
        self,
        embeds_size: int,
        gpu_id: int,
        reference_index=None,
        reference_nprobes=32,
        retrieval_index=None,
        retrieval_nprobes=32,
    ):
        """
        Initializes a new instance of FaissCPURetriever.

        Args:
            embeds_size (int): The dimension size of embeddings.
            gpu_id (int): Specifies GPU device ID if `use_gpu` is True.
                                    Default is -1 (no GPU device).
            reference_index (Faiss index): a Faiss index to be used to search reference embeddings. Should use inner product metric, as this is the distance metric we are operating in.
            reference_nprobes (int): See nprobes for Faiss indices.
            retrieval_index (Faiss index): a Faiss index to be used to search reference embeddings. Should use inner product metric, as this is the distance metric we are operating in.
            retrieval_nprobes (int): See nprobes for Faiss indices.
        """
        self.gpu_id = gpu_id
        self.use_gpu = True

        resources = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        if reference_index is None:
            # set default reference index to flatip
            cpu_reference_index = faiss.IndexFlatIP(embeds_size)

        else:
            if reference_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise Exception("FAISS retrieval index must use inner product metric!")
            if retrieval_index.d != embeds_size + 1:
                raise Exception(
                    f"Reference index must have embedding size {embeds_size}!"
                )
            cpu_reference_index = reference_index

        self.reference_index = faiss.index_cpu_to_gpu(
            resources, gpu_id, cpu_reference_index, co
        )
        self.reference_index.nprobe = reference_nprobes

        if retrieval_index is None:
            cpu_retrieval_index = faiss.IndexFlatIP(embeds_size + 1)
        else:
            if retrieval_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                raise Exception("FAISS retrieval index must use inner product metric!")
            if retrieval_index.d != embeds_size + 1:
                raise Exception(
                    f"Retrieval index must have embedding size {embeds_size + 1} due to added bias dimension!"
                )
            cpu_retrieval_index = retrieval_index

        self.retrieval_index = faiss.index_cpu_to_gpu(
            resources, gpu_id, cpu_retrieval_index, co
        )
        self.retrieval_index.nprobe = retrieval_nprobes

    def setup_retriever(
        self, retrieval_embeds, reference_embeds, alternate_ks, batch_size
    ):
        # train your indices here
        numpy_reference_embeds = reference_embeds.cpu().numpy()
        numpy_retrieval_embeds = retrieval_embeds.cpu().numpy()

        if not self.reference_index.is_trained:
            self.reference_index.train(numpy_reference_embeds)
        self.reference_index.add(numpy_reference_embeds)

        alignment_means = self.compute_alignment_means(
            retrieval_embeds, reference_embeds, alternate_ks, batch_size
        )
        modified_retrieval_embeds = np.concatenate(
            [numpy_retrieval_embeds, alignment_means], axis=1
        )
        if not self.retrieval_index.is_trained:
            self.retrieval_index.train(modified_retrieval_embeds)
        self.retrieval_index.add(modified_retrieval_embeds)

        return alignment_means

    def compute_alignment_means(
        self, retrieval_embeds, reference_embeds, alternate_ks, batch_size
    ):
        retrieval_embeds = retrieval_embeds.cpu().numpy()
        batch_reference_scores, indices = self.reference_index.search(
            retrieval_embeds, alternate_ks
        )
        return np.mean(batch_reference_scores, axis=-1, keepdims=True)

    def retrieve(
        self,
        retrieval_embeds,
        batch_query,
        top_k,
        alternate_weight,
        alignment_means,
        batch_size,
    ):
        # append -alt_weight to each vector in the query to account for the -alt_weight * reference_score term
        batch_query = batch_query.cpu().numpy()
        batch_query = np.concatenate(
            [
                batch_query,
                -alternate_weight
                * np.ones((batch_query.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        )
        distances, indices = self.retrieval_index.search(batch_query, top_k)
        return distances, indices
