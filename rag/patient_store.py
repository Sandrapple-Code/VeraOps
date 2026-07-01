import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Any, Union
from rag.embeddings import get_embedding

# Paths for saving the index and metadata inside the vector_store/patient_index/ directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_store", "patient_index", "patients.index")
METADATA_PATH = os.path.join(PROJECT_ROOT, "vector_store", "patient_index", "patients_metadata.pkl")

# In-memory stores
_index = None
# Holds list of dicts: [{"patient_id": "P001", "text": "..."}]
_patient_metadata: List[Dict[str, str]] = []

def get_patients_index() -> faiss.IndexFlatL2:
    """
    Returns the in-memory FAISS patients index, loading it from disk if available.
    If no index exists, a new L2 flat index of dimension 384 is initialized.
    """
    global _index, _patient_metadata
    if _index is None:
        db_dir = os.path.dirname(INDEX_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            _index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "rb") as f:
                _patient_metadata = pickle.load(f)
        else:
            _index = faiss.IndexFlatL2(384)
            _patient_metadata = []
    return _index

def clear_patient_store() -> None:
    """
    Clears the patient store from both memory and disk.
    """
    global _index, _patient_metadata
    _index = faiss.IndexFlatL2(384)
    _patient_metadata = []
    if os.path.exists(INDEX_PATH):
        try:
            os.remove(INDEX_PATH)
        except Exception:
            pass
    if os.path.exists(METADATA_PATH):
        try:
            os.remove(METADATA_PATH)
        except Exception:
            pass

def create_patient_text(patient_data: Dict[str, Any]) -> str:
    """
    Converts a patient structured dictionary into a formatted text summary.
    
    :param patient_data: Dictionary containing patient details.
    :return: A formatted summary string.
    """
    summary = f"""Patient ID: {patient_data.get('patient_id', 'N/A')}
Age: {patient_data.get('age', 'N/A')}
Gender: {patient_data.get('gender', 'N/A')}
Diagnosis: {patient_data.get('diagnosis', 'N/A')}
Symptoms: {patient_data.get('symptoms', 'N/A')}
Medicines: {patient_data.get('medicines', 'N/A')}
Ward: {patient_data.get('ward', 'N/A')}
Bed: {patient_data.get('bed_number', 'N/A')}
Notes: {patient_data.get('visit_notes', 'N/A')}"""
    return summary.strip()

def embed_patient(patient_text: str) -> np.ndarray:
    """
    Generates an embedding vector for the patient text summary.
    
    :param patient_text: Formatted patient text summary.
    :return: A 1D numpy array representing the embedding.
    """
    return get_embedding(patient_text)

def add_patient_vector(patient_id: str, vector: np.ndarray, patient_text: str = "") -> None:
    """
    Adds a patient embedding vector to the patient FAISS store and saves the index.
    
    :param patient_id: ID of the patient.
    :param vector: 1D or 2D numpy array representing the embedding vector.
    :param patient_text: The formatted text summary of the patient.
    """
    global _patient_metadata
    index = get_patients_index()
    
    # Standardize vector to 2D numpy array of shape (1, 384)
    vector_2d = np.array(vector, dtype=np.float32).reshape(1, -1)
    
    # Add vector to FAISS
    index.add(vector_2d)
    
    # Append mapping metadata
    _patient_metadata.append({
        "patient_id": patient_id,
        "text": patient_text
    })
    
    # Auto-save index to disk to ensure persistence
    save_patients_index()

def save_patients_index(index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH) -> None:
    """
    Saves the patient FAISS index and metadata to disk.
    
    :param index_path: Custom path to save the index. Defaults to INDEX_PATH.
    :param metadata_path: Custom path to save the metadata. Defaults to METADATA_PATH.
    """
    global _index, _patient_metadata
    if _index is not None:
        db_dir = os.path.dirname(index_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        faiss.write_index(_index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(_patient_metadata, f)

def search_similar_patients(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches the patient FAISS index for similar patients based on a text query.
    
    :param query: Query string describing clinical symptoms or patient profile.
    :param k: Number of similar patients to return.
    :return: List of dictionaries with 'patient_id', 'text', and 'distance'.
    """
    index = get_patients_index()
    if index.ntotal == 0:
        return []
        
    query_vector = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vector, min(k, index.ntotal))
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(_patient_metadata):
            meta = _patient_metadata[idx]
            results.append({
                "patient_id": meta["patient_id"],
                "text": meta["text"],
                "distance": float(dist)
            })
    return results

if __name__ == "__main__":
    from rag.utils import clear_index_files
    print("Testing Patient Store module...")
    
    # Ensure clean state for the test
    test_index = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_patients.index")
    test_meta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_patients_metadata.pkl")
    clear_index_files(test_index, test_meta)
    
    # Override global paths temporarily for testing
    original_idx, original_meta = INDEX_PATH, METADATA_PATH
    INDEX_PATH, METADATA_PATH = test_index, test_meta
    
    # Reset in-memory cache
    _index = None
    _patient_metadata = []
    
    try:
        patient_data = {
            "patient_id": "P001",
            "age": 45,
            "gender": "Male",
            "diagnosis": "Hypertension",
            "symptoms": "High blood pressure, headache",
            "medicines": "Lisinopril",
            "ward": "Cardiology",
            "bed_number": "C-01",
            "visit_notes": "Stressed due to work. Resting."
        }
        
        print("\nCreating patient summary...")
        text_summary = create_patient_text(patient_data)
        print("Summary:\n", text_summary)
        
        print("\nEmbedding patient...")
        vector = embed_patient(text_summary)
        assert vector.shape == (384,)
        print("Embedding shape:", vector.shape)
        
        print("\nAdding patient vector to store...")
        add_patient_vector(patient_data["patient_id"], vector, text_summary)
        
        print("Testing search (Query: 'High blood pressure Cardiology')...")
        results = search_similar_patients("High blood pressure Cardiology", k=1)
        print("Search results:")
        for r in results:
            print(f"- [Patient: {r['patient_id']}] [Distance: {r['distance']:.4f}]")
            print(f"Summary:\n{r['text']}")
            
        assert len(results) > 0
        assert results[0]["patient_id"] == "P001"
        print("\nPatient store checks passed successfully!")
        
    finally:
        # Restore paths
        INDEX_PATH, METADATA_PATH = original_idx, original_meta
        # Clean up files
        clear_index_files(test_index, test_meta)
        _index = None
        _patient_metadata = []
