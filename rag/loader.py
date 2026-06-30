import fitz  # PyMuPDF
import os

def load_pdf(file_path: str) -> str:
    """
    Loads a PDF file and extracts text page by page, returning the concatenated text.
    
    :param file_path: Absolute or relative path to the PDF file.
    :return: Concatenated text content extracted from the PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    doc = fitz.open(file_path)
    text_content = []
    
    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text:
                text_content.append(text)
    finally:
        doc.close()
        
    return "\n".join(text_content)
