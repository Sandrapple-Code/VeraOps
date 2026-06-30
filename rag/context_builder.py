from typing import List, Dict, Any, Optional

def build_context(
    docs: List[Dict[str, Any]], 
    patients: List[Dict[str, Any]], 
    structured_record: Optional[Dict[str, Any]] = None
) -> str:
    """
    Assembles retrieved documents, similar clinical cases, and optional structured patient record
    into a formatted context string.
    
    :param docs: List of document chunks.
    :param patients: List of patient similarity profiles.
    :param structured_record: Dictionary of the current patient's record.
    :return: Formatted text string containing all context.
    """
    context_parts = []
    
    # 1. Structured Patient Information
    if structured_record:
        context_parts.append("=== CURRENT PATIENT RECORD ===")
        for key, val in structured_record.items():
            field = key.replace("_", " ").title()
            context_parts.append(f"{field}: {val if val is not None else 'N/A'}")
        context_parts.append("==============================\n")
        
    # 2. Hospital Guidelines & SOPs
    if docs:
        context_parts.append("=== HOSPITAL GUIDELINES & SOPS ===")
        for idx, doc in enumerate(docs, 1):
            context_parts.append(f"Guideline Reference {idx}:")
            context_parts.append(doc.get("text", "").strip())
            context_parts.append("-" * 40)
        context_parts.append("==================================\n")
        
    # 3. Similar Patient Cases
    if patients:
        context_parts.append("=== SIMILAR CLINICAL CASES ===")
        for idx, pat in enumerate(patients, 1):
            context_parts.append(f"Case {idx} (Similarity Distance: {pat.get('distance', 0.0):.4f}):")
            context_parts.append(pat.get("text", "").strip())
            context_parts.append("-" * 40)
        context_parts.append("==============================\n")
        
    return "\n".join(context_parts)
