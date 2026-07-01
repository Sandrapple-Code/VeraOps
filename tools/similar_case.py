import json
from langchain_core.tools import tool
from rag.patient_store import search_similar_patients
from db.sqlite import get_patient

@tool
def similar_case_search(query: str, k: int = 2) -> str:
    """
    Finds and retrieves the most similar historical patient cases based on 
    clinical diagnosis and symptoms, returning both the FAISS summary text and 
    their structured database record.
    
    :param query: Symptoms or diagnosis description query string.
    :param k: Number of similar patient profiles to return. Defaults to 2.
    :return: JSON formatted string containing matching patient cases.
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."
        
    try:
        results = search_similar_patients(query.strip(), k=k)
        if not results:
            return "No similar historical cases found."
            
        compiled_cases = []
        for match in results:
            pid = match.get("patient_id")
            # Pull structured details from SQLite
            record = get_patient(pid)
            compiled_cases.append({
                "patient_id": pid,
                "distance": match.get("distance"),
                "summary": match.get("text"),
                "structured_record": dict(record) if record else None
            })
            
        return json.dumps(compiled_cases, indent=2)
    except Exception as e:
        return f"Error searching similar patient cases: {str(e)}"
