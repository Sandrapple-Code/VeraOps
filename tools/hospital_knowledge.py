import json
from langchain_core.tools import tool
from rag.document_store import search_docs

@tool
def hospital_knowledge_search(query: str, k: int = 3) -> str:
    """
    Searches the hospital FAISS vector database (containing SOPs, ICU protocols, 
    and emergency guidelines) for relevant document chunks.
    
    :param query: Medical or operational question query string.
    :param k: Number of relevant document chunks to return. Defaults to 3.
    :return: JSON formatted string of matching chunks with L2 distances.
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."
        
    try:
        results = search_docs(query.strip(), k=k)
        if not results:
            return "No relevant hospital guidelines or SOPs found."
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching hospital guidelines: {str(e)}"
