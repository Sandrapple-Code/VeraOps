import os
import sys
import json
from datetime import datetime
from langchain_core.tools import tool
from services.patient_service import register_new_patient

@tool
def register_patient(patient_data_json: str) -> str:
    """
    Registers a new patient in the hospital system.
    Inserts their record into the SQLite database, creates their document directory,
    generates initial markdown clinical reports, and embeds/indexes them in FAISS.
    
    :param patient_data_json: JSON string containing all patient attributes:
      patient_id (required), name (required), age (required), gender (required), 
      diagnosis, medicines, ward, bed_number, visit_notes, blood_group, phone_number,
      email, address, emergency_contact_name, emergency_contact_number, department,
      assigned_doctor, date_of_admission, admission_summary, current_status.
    :return: A success or error message string.
    """
    try:
        data = json.loads(patient_data_json)
    except Exception as e:
        return f"Error parsing input JSON: {str(e)}"
        
    try:
        res = register_new_patient(data)
        return res
    except Exception as e:
        return f"Error during patient registration: {str(e)}"
