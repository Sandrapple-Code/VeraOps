from typing import Optional, Dict, Any
from rag.retriever import retrieve_documents, retrieve_similar_patients, retrieve_patient_record
from rag.context_builder import build_context
from llm.groq_client import generate_response

def answer_query(api_key: str, query: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes the complete traditional RAG pipeline:
    1. Retrieve relevant hospital SOP document chunks
    2. Retrieve semantically similar patient records
    3. (Optional) Retrieve current patient's record from SQLite if patient_id is provided
    4. Build unified structured context
    5. Construct prompt and generate LLM response via Groq
    
    :param api_key: User-provided Groq API key.
    :param query: Medical or operational question.
    :param patient_id: Optional specific patient ID to fetch structured data.
    :return: Dictionary containing generated answer and intermediate retrieved context.
    """
    # 1. Retrieve guidelines/SOPs
    docs = retrieve_documents(query, k=3)
    
    # 2. Retrieve similar cases
    patients = retrieve_similar_patients(query, k=2)
    
    # 3. Retrieve structured patient record (if ID is provided)
    patient_record = None
    if patient_id and patient_id.strip():
        patient_record = retrieve_patient_record(patient_id.strip())
        
    # 4. Build context
    context = build_context(docs, patients, patient_record)
    
    # 5. Formulate instructions and prompt for LLM
    system_instruction = (
        "You are VeraOps, a helpful AI Hospital Assistant. You have access to hospital guidelines, "
        "similar clinical cases, and current patient records. Use these references to answer the query "
        "professionally and accurately. Cite guidelines and compare cases where helpful."
    )
    
    prompt = (
        f"{system_instruction}\n\n"
        f"=== RETRIEVED REFERENCE CONTEXT ===\n"
        f"{context}\n"
        f"====================================\n\n"
        f"User Query: {query}\n"
        f"Assistant Answer:"
    )
    
    # 6. Generate LLM response
    answer = generate_response(api_key, prompt)
    
    return {
        "answer": answer,
        "documents": docs,
        "patients": patients,
        "patient_record": patient_record
    }
