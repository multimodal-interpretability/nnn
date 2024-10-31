# nearest-neighbor-normalization
Simple and efficient training-free methods for correcting errors in contrastive image-text retrieval!


## Installation

For CPU-only:
```
pip install -e .
```

For Faiss GPU support, install the `gpu` extras:
```
pip install -e .[gpu]
```


## Example usage

Here's a demonstration of how to rerank CLIP embeddings using NNN:

```python
import numpy as np
from nnn import NNNRetriever, NNNRanker
from transformers import CLIPProcessor, CLIPModel
import torch

# Load the CLIP model and processor from Hugging Face
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Example images as PyTorch tensors (replace with your images)
images = [...]  # List of PIL Images or image file paths
image_inputs = processor(images=images, return_tensors="pt")

# Embed the images using CLIP
with torch.no_grad():
    image_embeddings = model.get_image_features(**image_inputs).cpu().numpy()

# Embed the caption text
caption = "A description of the images you want to match."
text_inputs = processor(text=[caption], return_tensors="pt")
with torch.no_grad():
    text_embedding = model.get_text_features(**text_inputs).cpu().numpy()

# Initialize the NNNRetriever and NNNRanker without additional reference points
nnn_retriever = NNNRetriever(image_embeddings.shape[1], use_gpu=False)  # Adjust dimensions if needed
nnn_ranker = NNNRanker(nnn_retriever, image_embeddings, image_embeddings, alternate_ks=2, batch_size=5, alternate_weight=0.75)

# Perform reranking using the text embedding as the query
_, indices = nnn_ranker.search(text_embedding, k=1)
print("Ranked image indices:", indices)
```
