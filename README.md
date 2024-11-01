# Nearest Neighbor Normalization Improves Multimodal Retrieval
### EMNLP 2024

<img align="right" width="38%" class="teaser" src="/assets/nnn_teaser.png">

### [ArXiv](https://arxiv.org/abs/2410.24114) | [Documentation](https://multimodal-interpretability.csail.mit.edu/nnn/)

[Neil Chowdhury](https://nchowdhury.com/)\*, [Franklin Wang](https://x.com/f_x_wang)\*, [Sumedh Shenoy](https://x.com/sumedhshenoy)\*, [Douwe Kiela](https://douwekiela.github.io/), [Sarah Schwettmann](https://cogconfluence.com/)†, [Tristan Thrush](http://www.tristanthrush.com/)†<br>
\*equal contribution †equal advising

Nearest Neighbor Normalization (NNN) is a simple and efficient training-free method for correcting errors in contrastive embedding-based retrieval!

By efficiently computing bias scores across each image in the retrieval database, NNN is able to consistently improve multimodal retrieval accuracy across a wide range of models and datasets. For instance, **we improve CLIP's image recall accuracy for MS-COCO by 7.1%!**

## Installation

You can install NNN directly with `pip` using 
```bash
pip install nnn-retrieval
```

For [Faiss](https://github.com/facebookresearch/faiss/) support (which significantly speeds up retrieval and retrieval dataset normalization calculations), follow the instructions [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) to install Faiss. NNN is compatible with both the CPU and GPU versions of Faiss.

For development, you can clone this repo locally, then install the package from source using:
```bash
pip install -e .[dev]
```

## Basic Usage

Here's how you can leverage NNN for text-to-image retrieval. To construct your retrieval database, you'll need:
- `image_embeddings`: Your database of image embeddings that you are retrieving from
- `reference_query_embeddings`: Your reference database of caption embeddings which NNN will use to compute the bias scores for each image embedding.
    - For example, this might be the training captions for the MS-COCO dataset if we are doing image retrieval with captions similar to MS-COCO.
    - Ideally, you should use a representative database of possible captions that are in-distribution to what you would see at inference time.

To instantiate the database and precompute the NNN bias scores, you can use the following code. The `image_embeddings` and `reference_query_embeddings` should be 2D NumPy arrays of shape `(|images|, embedding_dim)` and `(|reference_queries|, embedding_dim)`, respectively.

With GPU:
```python
from nnn import NNNRetriever, NNNRanker
nnn_retriever = NNNRetriever(image_embeddings.shape[1], use_gpu=True, gpu_id=0)
nnn_ranker = NNNRanker(nnn_retriever, image_embeddings, reference_embeddings, alternate_ks=128, alternate_weight=0.75, batch_size=256, use_gpu=True, gpu_id=0)
```

With CPU only:
```python
from nnn import NNNRetriever, NNNRanker
nnn_retriever = NNNRetriever(image_embeddings.shape[1])
nnn_ranker = NNNRanker(nnn_retriever, image_embeddings, reference_embeddings, alternate_ks=128, alternate_weight=0.75, batch_size=256)
```

The `alternate_ks` and `alternate_weight` arguments are hyperparameters for NNN. We recommend sweeping through these parameters to obtain the best results, but in general `alternate_ks=128` and `alternate_weight=0.75` works pretty well. See Appendix-B of the NNN paper for more information about hyperparameter sweeping.

Finally, to perform retrieval inference on a set of caption embeddings `text_embeddings` (also should be formatted as a 2D NumPy array), you can run:
```python
scores, indices = nnn_ranker.search(text_embeddings, top_k=5)
```

This will return the `top_k` highest retrieval scores and corresponding image indices for each caption embedding.

To use Faiss as the retrieval backend, simply swap the `NNNRetriever` for `FaissCPURetriever` or `FaissGPURetriever`.

## Full Examples

In [examples/nnn_clip_flickr30k.py](https://github.com/multimodal-interpretability/nnn/blob/main/examples/nnn_clip_flickr_30k.py), we also show a full end-to-end example of using NNN for image-to-text retrieval using the Flickr30k dataset and the [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) model. To install the additional dependencies for this example, you can run:
```bash
pip install transformers datasets
```
