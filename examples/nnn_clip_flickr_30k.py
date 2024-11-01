# In this script, we compare NNN to normal retrieval for the Flickr30k dataset and the CLIP model

from nnn import NNNRetriever, NNNRanker, BaseRetriever, BaseRanker

from datasets import load_dataset
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from torch.nn.functional import normalize

# NNN Hyperparameters:
nnn_k = 128
nnn_alpha = 0.75

def get_flickr30k_dataset():
    flickr30k = load_dataset("nlphuji/flickr30k")["test"]
    
    flickr30k_train = flickr30k.filter(lambda row: row['split'] == "train", num_proc=8)
    # get all the training captions, there's multiple captions per image so we flatten
    train_captions = [caption for caption_list in flickr30k_train["caption"] for caption in caption_list]

    # get the test set data: images + captions + ground truth labels
    flickr30k_test = flickr30k.filter(lambda row: row['split'] == "test", num_proc=8)

    test_images = flickr30k_test["image"]

    test_ground_truth_image_labels = []
    test_captions = []
    # for flickr30k, there will be ~5 captions for each image
    for i, caption_list in enumerate(flickr30k_test["caption"]):
        test_ground_truth_image_labels.extend([i] * len(caption_list))
        test_captions.extend(caption_list)

    return train_captions, test_images, test_captions, np.array(test_ground_truth_image_labels)

reference_captions, images, captions, ground_truth_image_labels = get_flickr30k_dataset()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with torch.inference_mode():
    # Embed the images using CLIP
    image_batch_size = 256

    image_embeddings = []

    for i in tqdm(range(0, len(images), image_batch_size)):
        image_inputs_batch = processor(images=images[i:i+image_batch_size], return_tensors="pt").to(device)
        image_embeddings_batch = normalize(model.get_image_features(**image_inputs_batch)).cpu().numpy() # move back to CPU for NNN
        image_embeddings.append(image_embeddings_batch)

    image_embeddings = np.concatenate(image_embeddings)

    # Create reference embeddings from in-distribution captions
    captions_batch_size = 256

    reference_embeddings = []
    
    for i in tqdm(range(0, len(reference_captions), captions_batch_size)):
        reference_inputs_batch = processor(text=reference_captions[i:i+captions_batch_size], return_tensors="pt", padding=True, truncation=True).to(device)
        reference_embeddings_batch = normalize(model.get_text_features(**reference_inputs_batch)).cpu().numpy()
        reference_embeddings.append(reference_embeddings_batch)

    reference_embeddings = np.concatenate(reference_embeddings)

# Initalize NNN retriever and ranker
# The NNNRanker will compute the bias scores
if device == "cuda":
    nnn_retriever = NNNRetriever(image_embeddings.shape[1], use_gpu=True, gpu_id=0)
    nnn_ranker = NNNRanker(nnn_retriever, image_embeddings, reference_embeddings, alternate_ks=nnn_k, alternate_weight=nnn_alpha, batch_size=1024, use_gpu=True, gpu_id=0)
else:
    nnn_retriever = NNNRetriever(image_embeddings.shape[1])
    nnn_ranker = NNNRanker(nnn_retriever, image_embeddings, reference_embeddings, alternate_ks=nnn_k, alternate_weight=nnn_alpha, batch_size=1024)

# Perform ranking using standard retrieval (no NNN corrections)
if device == "cuda":
    base_retriever = BaseRetriever(image_embeddings.shape[1], use_gpu=True, gpu_id=0)
    base_ranker = BaseRanker(nnn_retriever, image_embeddings, reference_embeddings, batch_size=1024, use_gpu=True, gpu_id=0)
else:
    base_retriever = BaseRetriever(image_embeddings.shape[1])
    base_ranker = BaseRanker(nnn_retriever, image_embeddings, reference_embeddings, batch_size=1024)

# Now we use NNN to do retrieval on the test set captions

with torch.inference_mode():
    captions_batch_size = 256
    caption_embeddings = []

    for i in tqdm(range(0, len(captions), captions_batch_size)):
        text_inputs_batch = processor(text=captions[i:i+captions_batch_size], return_tensors="pt", padding=True, truncation=True).to(device)
        caption_embedding_batch = normalize(model.get_text_features(**text_inputs_batch)).cpu().numpy()
        caption_embeddings.append(caption_embedding_batch)

    caption_embeddings = np.concatenate(caption_embeddings)

# get the k nearest embeddings for each test set caption using the NNN adjusted scores
k = 5
nnn_scores, nnn_indices = nnn_ranker.search(caption_embeddings, k)

# get the k nearest embeddings for each test set caption using the normal scores with no NNN adjustment
base_scores, base_indices = base_ranker.search(caption_embeddings, k)

for i in range(1, k + 1):
    nnn_correct_image_in_topi = np.any((nnn_indices == ground_truth_image_labels[:, None])[:, :i], axis=-1)
    base_correct_image_in_topi = np.any((base_indices == ground_truth_image_labels[:, None])[:, :i], axis=-1)
    print(f"NNN Test Set Retrieval Acc @ {i} = {np.mean(nnn_correct_image_in_topi)}")
    print(f"Normal Retrieval Test Set Retrieval Acc @ {i} = {np.mean(base_correct_image_in_topi)}")
