import json
from langchain_core.tools import tool
from db.sqlite import get_patient, get_connection

@tool
def patient_lookup(identifier: str) -> str:
    """
    Retrieves structured patient information from the SQLite database.
    Accepts either a unique Patient ID (e.g., 'P001') or a patient's full name.
    
    :param identifier: Unique Patient ID or Patient Full Name.
    :return: JSON formatted string containing matching patient records, or an error message.
    """
    if not identifier or not identifier.strip():
        return "Error: Patient identifier cannot be empty."
        
    identifier = identifier.strip()
    try:
        # 1. Search by Patient ID
        record = get_patient(identifier)
        if record:
            return json.dumps(dict(record), indent=2)
            
        # 2. Search by Name (case-insensitive substring match)
        conn = get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM patients WHERE name LIKE ?",
                (f"%{identifier}%",)
            )
            rows = cursor.fetchall()
            if rows:
                return json.dumps([dict(row) for row in rows], indent=2)
        finally:
            conn.close()
            
        return f"No patient found matching identifier: '{identifier}'"
    except Exception as e:
        return f"Error occurred during patient lookup: {str(e)}"
