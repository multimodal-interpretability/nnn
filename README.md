# Nearest Neighbor Normalization (EMNLP 2024)
Nearest Neighbor Normalization (NNN) is a simple and efficient training-free method for correcting errors in contrastive embedding-based retrieval!

## Installation

You can install NNN directly with `pip` using 
```
pip install -e .
```

For [Faiss](https://github.com/facebookresearch/faiss/) support (which significantly speeds up retrieval and retrieval dataset normalization calculations), follow the installation instructions [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md). NNN is compatible with both the CPU and GPU versions of Faiss.

For development, you can clone this repo locally, then install the package using:
```
pip install -e .[dev]
```

## Example usage

Here's a demonstration of how to rerank CLIP embeddings using NNN. This is basic usage; for deployment, consider using a Faiss-based retriever for better performance (e.g. `FaissGPURetriever`).

To run this example, you'll need to install `transformers`, `pillow`, and `requests`.

```python
import numpy as np
from nnn import NNNRetriever, NNNRanker
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Example images as PyTorch tensors (replace with your images)
image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/0/00/St_Louis_night_expblend_cropped.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/da/The_Parthenon_in_Athens.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/48/Alabamahills.jpg"
]

images = [Image.open(BytesIO(requests.get(url, headers={'User-Agent': 'curl/7.64.1'}).content)) for url in image_urls]
image_inputs = processor(images=images, return_tensors="pt").to(device)

# Embed the images using CLIP
with torch.no_grad():
    image_embeddings = model.get_image_features(**image_inputs).cpu().numpy()  # move back to CPU for NNN

# Embed the caption text (used as an input for retrieval)
caption = "A description of the images you want to match."
text_inputs = processor(text=[caption], return_tensors="pt").to(device)
with torch.no_grad():
    text_embedding = model.get_text_features(**text_inputs).cpu().numpy()

# Create reference embeddings from in-distribution captions
reference_captions = [f"Reference caption {i}" for i in range(1, 11)]
reference_inputs = processor(text=reference_captions, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    reference_embeddings = model.get_text_features(**reference_inputs).cpu().numpy()

# Perform ranking using NNN
if device == "cuda":
    nnn_retriever = NNNRetriever(image_embeddings.shape[1], use_gpu=True, gpu_id=0)
    nnn_ranker = NNNRanker(nnn_retriever, image_embeddings, reference_embeddings, alternate_ks=8, batch_size=8, use_gpu=True, gpu_id=0)
else:
    nnn_retriever = NNNRetriever(image_embeddings.shape[1])
    nnn_ranker = NNNRanker(nnn_retriever, image_embeddings, reference_embeddings, alternate_ks=8, batch_size=8)

_, indices = nnn_ranker.search(text_embedding, 5)
print("Ranked image indices:", indices)
# Ranked image indices: [[0 4 2 3 1]]
```
