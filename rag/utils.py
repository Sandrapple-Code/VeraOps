import os

def clear_index_files(index_path: str, metadata_path: str) -> None:
    """
    Helper to delete index and metadata files from disk for testing/resetting purposes.
    
    :param index_path: Path to the FAISS index file.
    :param metadata_path: Path to the pickle metadata file.
    """
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
        except Exception as e:
            print(f"Error removing index file {index_path}: {e}")
            
    if os.path.exists(metadata_path):
        try:
            os.remove(metadata_path)
        except Exception as e:
            print(f"Error removing metadata file {metadata_path}: {e}")
