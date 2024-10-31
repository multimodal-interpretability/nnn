import numpy as np
from nnn import BaseRetriever
from nnn import BaseRanker
from nnn import NNNRetriever
from nnn import NNNRanker
import math
import pytest


@pytest.mark.not_faiss  # This test does not depend on FAISS
def test_exhaustive_search():
    square_retrieval_points = np.array(
        [[0, 1.0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0]],
        dtype=np.float32,
    )

    jitter = 0.01
    square_query_points = np.array(
        [
            [0, 0.6, 0.8, 0],
            [0.6, 0, 0.8, 0],
            [0, -0.6, 0.8, 0],
            [-0.6, 0, 0.8, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )

    square_reference_points = np.concatenate(
        (
            square_query_points + np.array([0, jitter, 0, 0], dtype=np.float32),
            square_query_points + np.array([0, -jitter, 0, 0], dtype=np.float32),
            square_query_points + np.array([jitter, 0, 0, 0], dtype=np.float32),
            square_query_points + np.array([-jitter, 0, 0, 0], dtype=np.float32),
        ),
        axis=0,
    )  # jitter to create reference points (same distribution)

    base_retriever = BaseRetriever(4, False)
    nnn_retriever = NNNRetriever(4, False)

    fake_ranker = BaseRanker(
        base_retriever, square_retrieval_points, square_reference_points, batch_size=5
    )  # dummy ranker w/ no weights should return original search results!
    true_ranker = NNNRanker(
        nnn_retriever,
        square_retrieval_points,
        square_reference_points,
        alternate_ks=2,
        batch_size=5,
        alternate_weight=0.75,
    )  # true ranker w/ weight

    old_scores, old_indices = fake_ranker.search(square_query_points, 1)
    old_scores = old_scores.flatten().tolist()
    old_indices = old_indices.flatten().tolist()

    scores, indices = true_ranker.search(square_query_points, 1)
    scores = scores.flatten().tolist()
    indices = indices.flatten().tolist()

    gt_old_indices = [4, 4, 4, 4, 4]
    gt_old_scores = [0.8, 0.8, 0.8, 0.8, 1.0]
    gt_new_scores = [0.14625, 0.14625, 0.14625, 0.14625, 0.25]
    gt_new_indices = [0, 1, 2, 3, 4]

    for i in range(5):
        assert old_indices[i] == gt_old_indices[i]
        assert old_scores[i] == pytest.approx(gt_old_scores[i], abs=2e-5)
        assert indices[i] == gt_new_indices[i]
        assert scores[i] == pytest.approx(gt_new_scores[i], abs=2e-5)
    print("passed exhaustive search!")


@pytest.mark.faiss_cpu  # This test is for FAISS-CPU
def test_faiss_cpu_search():
    import faiss
    from nnn import FaissCPURetriever

    square_retrieval_points = np.array(
        [[0, 1.0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0]],
        dtype=np.float32,
    )

    jitter = 0.01
    square_query_points = np.array(
        [
            [0, 0.6, 0.8, 0],
            [0.6, 0, 0.8, 0],
            [0, -0.6, 0.8, 0],
            [-0.6, 0, 0.8, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )

    square_reference_points = np.concatenate(
        (
            square_query_points + np.array([0, jitter, 0, 0], dtype=np.float32),
            square_query_points + np.array([0, -jitter, 0, 0], dtype=np.float32),
            square_query_points + np.array([jitter, 0, 0, 0], dtype=np.float32),
            square_query_points + np.array([-jitter, 0, 0, 0], dtype=np.float32),
        ),
        axis=0,
    )  # jitter to create reference points (same distribution)

    nnn_retriever = FaissCPURetriever(4)
    true_ranker = NNNRanker(
        nnn_retriever,
        square_retrieval_points,
        square_reference_points,
        alternate_ks=2,
        batch_size=5,
        alternate_weight=0.75,
    )  # true ranker w/ weight

    scores, indices = true_ranker.search(square_query_points, 1)
    scores = scores.flatten().tolist()
    indices = indices.flatten().tolist()

    gt_new_scores = [0.14625, 0.14625, 0.14625, 0.14625, 0.25]
    gt_new_indices = [0, 1, 2, 3, 4]

    for i in range(5):
        assert indices[i] == gt_new_indices[i]
        assert scores[i] == pytest.approx(gt_new_scores[i], abs=2e-5)
    print("passed faiss cpu!")


@pytest.mark.faiss_gpu  # This test is for FAISS-GPU
def test_faiss_gpu_search():
    import faiss
    from nnn import FaissGPURetriever

    square_retrieval_points = np.array(
        [[0, 1.0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0]],
        dtype=np.float32,
    )

    jitter = 0.01
    square_query_points = np.array(
        [
            [0, 0.6, 0.8, 0],
            [0.6, 0, 0.8, 0],
            [0, -0.6, 0.8, 0],
            [-0.6, 0, 0.8, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )

    square_reference_points = np.concatenate(
        (
            square_query_points + np.array([0, jitter, 0, 0], dtype=np.float32),
            square_query_points + np.array([0, -jitter, 0, 0], dtype=np.float32),
            square_query_points + np.array([jitter, 0, 0, 0], dtype=np.float32),
            square_query_points + np.array([-jitter, 0, 0, 0], dtype=np.float32),
        ),
        axis=0,
    )  # jitter to create reference points (same distribution)

    nnn_retriever = FaissGPURetriever(4, gpu_id=0)
    true_ranker = NNNRanker(
        nnn_retriever,
        square_retrieval_points,
        square_reference_points,
        alternate_ks=2,
        batch_size=5,
        alternate_weight=0.75,
        use_gpu=True,
        gpu_id=0,
    )  # true ranker w/ weight

    scores, indices = true_ranker.search(square_query_points, 1)
    scores = scores.flatten().tolist()
    indices = indices.flatten().tolist()

    gt_new_scores = [0.14625, 0.14625, 0.14625, 0.14625, 0.25]
    gt_new_indices = [0, 1, 2, 3, 4]

    for i in range(5):
        assert indices[i] == gt_new_indices[i]
        assert scores[i] == pytest.approx(gt_new_scores[i], abs=2e-5)
    print("passed faiss gpu!")


"""
if __name__ == "__main__":
    test_exhaustive_search()
    try:
        import faiss
        print(hasattr(faiss, 'StandardGpuResources'))
        if hasattr(faiss, 'StandardGpuResources'):
            print("has faiss gpu installed")
            test_faiss_gpu_search()
        else:
            test_faiss_cpu_search()
    except:
        print("faiss not installed!")
"""
