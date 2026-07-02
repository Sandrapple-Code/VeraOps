import os
import json
import glob
import fitz # PyMuPDF
from typing import Dict, Any, List, Callable, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import load_pdf, load_md
from rag.document_store import (
    create_doc_embeddings,
    save_faiss_index,
    get_docs_index,
    clear_document_store,
    INDEX_PATH as DOCS_INDEX_PATH
)
from rag.patient_store import (
    add_patient_vector,
    embed_patient,
    save_patients_index,
    get_patients_index,
    clear_patient_store,
    INDEX_PATH as PATIENTS_INDEX_PATH
)
from rag.embeddings import get_embedding_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSPITAL_DOCS_DIR = os.path.join(PROJECT_ROOT, "hospital_docs")
PATIENT_DOCS_DIR = os.path.join(PROJECT_ROOT, "patient_documents")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "vector_store", "ingestion_manifest.json")

def get_all_hospital_files() -> List[str]:
    """Finds all PDF and Markdown files in hospital_docs/."""
    files = []
    if os.path.exists(HOSPITAL_DOCS_DIR):
        for ext in ("*.pdf", "*.md"):
            files.extend(glob.glob(os.path.join(HOSPITAL_DOCS_DIR, ext)))
            files.extend(glob.glob(os.path.join(HOSPITAL_DOCS_DIR, "**", ext), recursive=True))
    return sorted(list(set(files)))

def get_all_patient_files() -> List[str]:
    """Finds all Markdown files in patient_documents/ recursively."""
    files = []
    if os.path.exists(PATIENT_DOCS_DIR):
        files.extend(glob.glob(os.path.join(PATIENT_DOCS_DIR, "**", "*.md"), recursive=True))
    return sorted(list(set(files)))

def get_file_metadata(filepath: str) -> Dict[str, Any]:
    """Returns size and modification time for a file."""
    try:
        stat = os.stat(filepath)
        return {
            "mtime": stat.st_mtime,
            "size": stat.st_size
        }
    except Exception:
        return {"mtime": 0, "size": 0}

def build_current_manifest() -> Dict[str, Dict[str, Any]]:
    """Builds a manifest of all files currently on disk."""
    manifest = {"hospital_docs": {}, "patient_docs": {}}
    for f in get_all_hospital_files():
        rel_path = os.path.relpath(f, PROJECT_ROOT)
        manifest["hospital_docs"][rel_path] = get_file_metadata(f)
    for f in get_all_patient_files():
        rel_path = os.path.relpath(f, PROJECT_ROOT)
        manifest["patient_docs"][rel_path] = get_file_metadata(f)
    return manifest

def check_needs_rebuild() -> bool:
    """
    Checks if the vector databases need to be rebuilt.
    Returns True if index files are missing, or if files have changed or been added/deleted.
    """
    # 1. Check if index files exist
    if not os.path.exists(DOCS_INDEX_PATH) or not os.path.exists(PATIENTS_INDEX_PATH):
        return True
        
    # 2. Check if manifest exists
    if not os.path.exists(MANIFEST_PATH):
        return True
        
    try:
        with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
            old_manifest = json.load(f)
    except Exception:
        return True
        
    # 3. Compare with current manifest
    current_manifest = build_current_manifest()
    return current_manifest != old_manifest

def rebuild_knowledge_base(progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[int, int]:
    """
    Rebuilds the hospital documents and patients FAISS indices from scratch.
    
    :param progress_callback: Optional callback receiving (percentage, status_text)
    :return: A tuple of (hospital_chunks_count, patient_chunks_count)
    """
    # Ensure directory existence
    os.makedirs(HOSPITAL_DOCS_DIR, exist_ok=True)
    os.makedirs(PATIENT_DOCS_DIR, exist_ok=True)
    
    # 1. Clear existing stores
    if progress_callback:
        progress_callback(0.0, "Resetting vector stores...")
    clear_document_store()
    clear_patient_store()
    
    # 2. Scan files
    h_files = get_all_hospital_files()
    p_files = get_all_patient_files()
    
    total_files = len(h_files) + len(p_files)
    processed_files = 0
    
    # Initialize LangChain RecursiveCharacterSplitter
    from services.config_service import get_setting
    c_size = get_setting("chunk_size", 1000)
    c_overlap = get_setting("chunk_overlap", 200)
    splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    
    hospital_chunks_count = 0
    patient_chunks_count = 0
    
    # 3. Process Hospital Documents
    for f in h_files:
        filename = os.path.basename(f)
        if progress_callback:
            percent = (processed_files / max(1, total_files)) * 0.9  # Keep 10% for final saving
            progress_callback(percent, f"Indexing Hospital Doc: {filename}...")
            
        try:
            if f.lower().endswith(".pdf"):
                doc = fitz.open(f)
                try:
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        page_text = page.get_text()
                        if page_text.strip():
                            chunks = splitter.split_text(page_text)
                            if chunks:
                                create_doc_embeddings(chunks, source=f, page=page_num + 1)
                                hospital_chunks_count += len(chunks)
                finally:
                    doc.close()
            elif f.lower().endswith(".md"):
                text = load_md(f)
                if text.strip():
                    chunks = splitter.split_text(text)
                    if chunks:
                        create_doc_embeddings(chunks, source=f, page=None)
                        hospital_chunks_count += len(chunks)
            else:
                processed_files += 1
                continue
        except Exception as e:
            print(f"Error processing hospital document {f}: {e}")
            
        processed_files += 1

    # 4. Process Patient Documents
    for f in p_files:
        filename = os.path.basename(f)
        parent_dir = os.path.dirname(f)
        patient_id = os.path.basename(parent_dir)
        
        if progress_callback:
            percent = (processed_files / max(1, total_files)) * 0.9
            progress_callback(percent, f"Indexing Patient Doc: {patient_id}/{filename}...")
            
        try:
            text = load_md(f)
            if text.strip():
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    if chunk.strip():
                        # Embed chunk and register with patient store
                        vector = embed_patient(chunk)
                        add_patient_vector(patient_id, vector, chunk, source=filename)
                        patient_chunks_count += 1
        except Exception as e:
            print(f"Error processing patient document {f}: {e}")
            
        processed_files += 1

    # 5. Save indices to disk
    if progress_callback:
        progress_callback(0.95, "Saving vector indices to disk...")
        
    save_faiss_index()
    save_patients_index()
    
    # 6. Save manifest
    current_manifest = build_current_manifest()
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(current_manifest, f, indent=2)
        
    if progress_callback:
        progress_callback(1.0, f"Indexing complete: {hospital_chunks_count} hospital chunks and {patient_chunks_count} patient chunks.")
        
    return hospital_chunks_count, patient_chunks_count

def get_knowledge_base_stats() -> Dict[str, Any]:
    """
    Returns metrics and metadata of the current RAG knowledge base.
    """
    # Force load indexes to get counts
    docs_idx = get_docs_index()
    patients_idx = get_patients_index()
    
    # Extract model name
    try:
        model = get_embedding_model()
        model_name = model.get_submodule("0").auto_model.config._name_or_path
    except Exception:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
    return {
        "hospital_docs_count": len(get_all_hospital_files()),
        "patient_docs_count": len(get_all_patient_files()),
        "total_chunks": docs_idx.ntotal + patients_idx.ntotal,
        "embedding_model": model_name,
        "hospital_vector_count": docs_idx.ntotal,
        "patient_vector_count": patients_idx.ntotal
    }

def index_patient_documents(patient_id: str) -> None:
    """
    Incrementally builds/rebuilds FAISS index vector chunks and manifest entries for a single patient's markdown files.
    """
    # 1. Remove existing vectors from FAISS
    from rag.patient_store import remove_patient_vectors, embed_patient, add_patient_vector, save_patients_index
    remove_patient_vectors(patient_id)
    
    # 2. Splitter settings
    from services.config_service import get_setting
    c_size = get_setting("chunk_size", 1000)
    c_overlap = get_setting("chunk_overlap", 200)
    splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    
    # 3. Locate files
    patient_dir = os.path.join(PATIENT_DOCS_DIR, patient_id)
    if not os.path.exists(patient_dir):
        return
        
    p_files = glob.glob(os.path.join(patient_dir, "*.md"))
    for f in p_files:
        try:
            text = load_md(f)
            if text.strip():
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    if chunk.strip():
                        vector = embed_patient(chunk)
                        add_patient_vector(patient_id, vector, chunk, source=os.path.basename(f))
        except Exception as e:
            print(f"Error incrementally indexing patient file {f}: {e}")
            
    # 4. Save patients index to disk
    save_patients_index()
    
    # 5. Sync/update manifest
    manifest = {"hospital_docs": {}, "patient_docs": {}}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except Exception:
            pass
            
    # Remove existing manifest entries for this patient
    prefix = f"patient_documents/{patient_id}/"
    to_remove = [k for k in manifest.get("patient_docs", {}) if k.replace("\\", "/").startswith(prefix)]
    for k in to_remove:
        del manifest["patient_docs"][k]
        
    # Scan current patient documents for this patient and add to manifest
    for f in p_files:
        rel_path = os.path.relpath(f, PROJECT_ROOT)
        manifest["patient_docs"][rel_path] = get_file_metadata(f)
        
    # Save updated manifest
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    try:
        with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"Failed to save manifest in incremental indexing: {e}")
