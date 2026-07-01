from typing import List, Dict, Any, Optional
from rag.document_store import search_docs
from rag.patient_store import search_similar_patients
from db.sqlite import get_patient

def retrieve_documents(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieves top-k document chunks from the hospital guidelines (SOPs) FAISS index.
    
    :param query: Query string.
    :param k: Number of chunks to retrieve.
    :return: List of dicts with 'text' and 'distance'.
    """
    if k is None:
        from services.config_service import get_setting
        k = get_setting("top_k_retrieval", 3)
    return search_docs(query, k=k)

def retrieve_similar_patients(query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieves top-k semantically similar historical patient profiles from the patient FAISS store.
    
    :param query: Clinical details/symptoms query string.
    :param k: Number of cases to retrieve.
    :return: List of dicts with 'patient_id', 'text', and 'distance'.
    """
    if k is None:
        from services.config_service import get_setting
        k = get_setting("top_k_retrieval", 2)
    return search_similar_patients(query, k=k)

def retrieve_patient_record(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific patient's structured record from SQLite.
    Does not perform embedding or similarity search.
    
    :param patient_id: Unique patient ID.
    :return: Dictionary representation of the patient record, or None.
    """
    record = get_patient(patient_id)
    return dict(record) if record else None
