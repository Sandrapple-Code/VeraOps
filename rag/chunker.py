from typing import List

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Splits text into chunks of specified chunk_size with an overlap of chunk_overlap.
    
    :param text: Text string to chunk.
    :param chunk_size: Maximum character count per chunk.
    :param chunk_overlap: Overlapping character count between consecutive chunks.
    :return: List of text chunks.
    """
    if not text:
        return []
        
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size.")
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Advance by chunk_size minus overlap
        start += (chunk_size - chunk_overlap)
        
        # Safety check to prevent infinite loop if overlap equals size
        if chunk_size - chunk_overlap <= 0:
            break
            
    return chunks
