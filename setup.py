from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nnn-retrieval", 
    version="0.1",
    packages=find_packages(),       
    install_requires=[              # Core dependencies
        'torch',                    
        'tqdm',
        'numpy',                    # Numerical computations
        'faiss-cpu',                # Potentially moving to optional dependency...
        'faiss-gpu-cu11'
    ],
    extras_require={
        'faiss-gpu': ['faiss-gpu-cu11'] 
    },
    author="Neil Chowdhury, Sumedh Shenoy, and Franklin Wang",
    description="Simple and efficient training-free methods for correcting errors in contrastive image-text retrieval!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sumedh-shenoy/nearest-neighbor-normalization",
    classifiers=[
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        # License
        "License :: OSI Approved :: MIT License",
        # Operating System
        "Operating System :: OS Independent",
        # Programming Language
        "Programming Language :: Python :: 3",
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
