import numpy as np
from nnn import BaseRetriever, BaseRanker, NNNRetriever, NNNRanker
import math


"""
This is our set of retrieval embeddings.
For the sake of this example, we take 5 points on the 4D unit hypersphere.
"""
square_retrieval_points = np.array([
    [0, 1.0, 0, 0],
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [-1, 0,  0, 0],
    [0, 0, 1, 0]
], dtype=np.float32)

"""
Query embeddings we want to use to retrieve.
"""
square_query_points = np.array([
    [0, 0.6, 0.8, 0],
    [0.6, 0, 0.8, 0],
    [0, -0.6, 0.8, 0],
    [-0.6, 0, 0.8, 0],
    [0, 0, 1, 0]
], dtype=np.float32)

jitter = 0.01

"""
This is our reference dataset, used to calculate debiasing scores. Note that the assumption
in our paper is that the reference set is drawn from a similar distribution as the query set. 
"""
square_reference_points = np.concatenate((square_query_points + np.array([0, jitter, 0, 0], dtype=np.float32), 
                                            square_query_points + np.array([0, -jitter, 0, 0], dtype=np.float32), 
                                            square_query_points + np.array([jitter, 0, 0, 0], dtype=np.float32), 
                                            square_query_points + np.array([-jitter, 0, 0, 0], dtype=np.float32)),
                                            axis = 0) # jitter to create reference points (same distribution)


"""
Here, we create two retriever modules. One using no modifications to the embedding scores, and operates as a standard maximum score retriever (BaseRetriever),
and one that uses the score-correction algorithm proposed in our paper, NNN. We pass in the size of the embeddings, as well as whether to use the GPU or not (and
what device to use if so). 

We also implement two FAISS-based retrievers that support our NNN score correction algorithm.
"""
base_retriever = BaseRetriever(4, False)
nnn_retriever = NNNRetriever(4, True)


"""
Create a ranker module, which does the actual heavy lifting——calculates the bias scores, and passes this information to the retriever
that is passed in at instantiation time. Supports different batch sizes (although this may be refactored to be hidden from user), and required
parameters for the different scoring algorithms (alernate_k and alternate_weight for NNN, and lambda for distribution normalization (see https://arxiv.org/pdf/2302.11084.))

Requires retrieval set + reference set to be passed in at instantation time. 
"""
base_ranker = BaseRanker(base_retriever, square_retrieval_points, square_reference_points, batch_size=5)                                           # dummy ranker w/ no weights should return original search results!
nnn_ranker = NNNRanker(nnn_retriever, square_retrieval_points, square_reference_points, alternate_ks=2, batch_size=5, alternate_weight=0.75)      # true ranker w/ weight      

"""
Search w/ query embeddings. 
"""
old_scores, old_indices = base_ranker.search(square_query_points, 1)
scores, indices = nnn_ranker.search(square_query_points, 1)
