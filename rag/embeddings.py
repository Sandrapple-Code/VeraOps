from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# In-memory cache for the model
_model = None

def get_embedding_model() -> SentenceTransformer:
    """
    Initializes and returns the shared sentence-transformer model.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts using the shared model.
    
    :param texts: List of text strings.
    :return: A numpy array of shape (len(texts), embedding_dim) in float32.
    """
    model = get_embedding_model()
    # Ensure float32 format as required by FAISS
    return model.encode(texts, convert_to_numpy=True).astype(np.float32)

def get_embedding(text: str) -> np.ndarray:
    """
    Generates embedding for a single text.
    
    :param text: Input text string.
    :return: A 1D numpy array representing the embedding.
    """
    return get_embeddings([text])[0]
