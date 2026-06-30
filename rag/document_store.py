import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Any
from rag.embeddings import get_embeddings, get_embedding

# Paths for saving the index and metadata inside the rag/ directory
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs.index")
METADATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs_metadata.pkl")

# In-memory store for the current index and chunks
_index = None
_chunks: List[str] = []

def get_docs_index() -> faiss.IndexFlatL2:
    """
    Returns the in-memory FAISS index, loading it from disk if available.
    If no index exists, a new L2 flat index of dimension 384 is initialized.
    """
    global _index, _chunks
    if _index is None:
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            _index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "rb") as f:
                _chunks = pickle.load(f)
        else:
            # sentence-transformers/all-MiniLM-L6-v2 output dimension is 384
            _index = faiss.IndexFlatL2(384)
            _chunks = []
    return _index

def create_doc_embeddings(chunks: List[str]) -> None:
    """
    Generates embeddings for chunks and adds them to the document FAISS index.
    
    :param chunks: List of text chunks.
    """
    if not chunks:
        return
        
    global _chunks
    index = get_docs_index()
    
    # Generate embeddings (must be float32 for FAISS)
    embeddings = get_embeddings(chunks)
    
    # Add to FAISS index
    index.add(embeddings)
    
    # Track the metadata chunks
    _chunks.extend(chunks)

def save_faiss_index(index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH) -> None:
    """
    Saves the document FAISS index and chunk metadata to disk.
    
    :param index_path: Custom path to save the index. Defaults to INDEX_PATH.
    :param metadata_path: Custom path to save the metadata. Defaults to METADATA_PATH.
    """
    global _index, _chunks
    if _index is not None:
        faiss.write_index(_index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(_chunks, f)

def search_docs(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches the FAISS document index for the query.
    
    :param query: Query string.
    :param k: Number of top results to return.
    :return: List of dictionaries containing 'text' and 'distance'.
    """
    index = get_docs_index()
    if index.ntotal == 0:
        return []
        
    # Generate query embedding vector
    query_vector = get_embedding(query).reshape(1, -1)
    
    # Perform search in index
    distances, indices = index.search(query_vector, min(k, index.ntotal))
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(_chunks):
            results.append({
                "text": _chunks[idx],
                "distance": float(dist)
            })
    return results

if __name__ == "__main__":
    from rag.utils import clear_index_files
    print("Testing Document Store module...")
    
    # Ensure a clean state for the test
    test_index = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_docs.index")
    test_meta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_docs_metadata.pkl")
    clear_index_files(test_index, test_meta)
    
    # Override global paths temporarily for testing
    original_idx, original_meta = INDEX_PATH, METADATA_PATH
    INDEX_PATH, METADATA_PATH = test_index, test_meta
    
    # Reset in-memory cache
    _index = None
    _chunks = []
    
    try:
        sample_chunks = [
            "Protocol for ICU admission requires checking patient vital signs and heart rate.",
            "Emergency SOP: Code Red indicates a fire emergency. Evacuate through nearest exit.",
            "General ward discharge checklist requires physician sign-off and pharmacy clearance."
        ]
        
        print("\nAdding document chunks to store...")
        create_doc_embeddings(sample_chunks)
        
        print("Saving index...")
        save_faiss_index()
        
        print("Testing search (Query: 'ICU vitals check')...")
        results = search_docs("ICU vitals check", k=2)
        print("Search results:")
        for r in results:
            print(f"- [Distance: {r['distance']:.4f}] {r['text']}")
            
        assert len(results) > 0
        assert "ICU admission" in results[0]["text"]
        print("\nDocument store checks passed successfully!")
        
    finally:
        # Restore paths
        INDEX_PATH, METADATA_PATH = original_idx, original_meta
        # Clean up files
        clear_index_files(test_index, test_meta)
        _index = None
        _chunks = []
