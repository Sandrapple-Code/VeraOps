import json
from langchain_core.tools import tool
from rag.patient_store import search_similar_patients

@tool
def patient_history_search(query: str, k: int = 3) -> str:
    """
    Searches the patient FAISS vector database to retrieve specific clinical segments
    and patient notes matching the query.
    
    :param query: Symptoms, diagnosis, or clinical notes query string.
    :param k: Number of text chunks to retrieve. Defaults to 3.
    :return: JSON formatted string of matching patient clinical history segments.
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."
        
    try:
        results = search_similar_patients(query.strip(), k=k)
        if not results:
            return "No matching patient clinical history records found."
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching patient clinical history: {str(e)}"
