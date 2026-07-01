import os
import sys
import json
from datetime import datetime
from langchain_core.tools import tool
from db.sqlite import update_patient, get_patient

# Add parent path to import generators
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.seed_database import (
    generate_patient_summary,
    generate_admission_report,
    generate_doctor_notes,
    generate_prescription,
    generate_lab_report,
    generate_radiology_report,
    generate_discharge_summary
)

@tool
def modify_patient_record(patient_id: str, updates_json: str) -> str:
    """
    Updates an existing patient's clinical and administrative details in the SQLite database
    and synchronizes their local Markdown clinical documents.
    
    :param patient_id: Unique Patient ID (e.g., 'P001').
    :param updates_json: JSON string containing fields to update (e.g. {"ward": "ICU", "bed_number": "ICU-02"}).
    :return: A success or error message string.
    """
    if not patient_id or not patient_id.strip():
        return "Error: patient_id is required."
        
    patient_id = patient_id.strip()
    try:
        updates = json.loads(updates_json)
    except Exception as e:
        return f"Error parsing updates JSON: {str(e)}"
        
    try:
        # 1. Fetch current patient details
        record = get_patient(patient_id)
        if not record:
            return f"Error: Patient with ID {patient_id} does not exist."
            
        # 2. Update SQLite
        db_success = update_patient(patient_id, updates)
        if not db_success:
            return f"Info: No database changes applied for Patient ID {patient_id}."
            
        # 3. Retrieve the fully updated patient record
        updated_record = get_patient(patient_id)
        
        # 4. Regenerate Markdown clinical files to keep them synchronized
        doc_dir = os.path.join(PROJECT_ROOT, "patient_documents", patient_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Merge existing updates to rebuild JSON format for template generators
        gen_data = {
            "Patient ID": patient_id,
            "Full Name": updated_record.get("name"),
            "Age": updated_record.get("age"),
            "Gender": updated_record.get("gender"),
            "Blood Group": updates.get("blood_group", "Unknown"),
            "Phone Number": updates.get("phone_number", "N/A"),
            "Email": updates.get("email", "N/A"),
            "Address": updates.get("address", "N/A"),
            "Emergency Contact Name": updates.get("emergency_contact_name", "N/A"),
            "Emergency Contact Number": updates.get("emergency_contact_number", "N/A"),
            "Department": updates.get("department", "General Medicine"),
            "Assigned Doctor": updates.get("assigned_doctor", "Staff Physician"),
            "Ward": updated_record.get("ward"),
            "Bed Number": updated_record.get("bed_number"),
            "Disease / Diagnosis": updated_record.get("diagnosis"),
            "Symptoms": updates.get("symptoms", updated_record.get("diagnosis")),
            "Past Medical History": updates.get("past_medical_history", "None"),
            "Allergies": updates.get("allergies", "NKDA"),
            "Current Medications": updates.get("current_medications", "None"),
            "Treatment Plan": updates.get("treatment_plan", "Standard supportive care."),
            "Current Status": updates.get("current_status", "Admitted"),
            "Admission Summary": updates.get("admission_summary", "Patient admitted."),
            "Doctor Notes": updated_record.get("visit_notes"),
            "Prescription": updated_record.get("medicines"),
            "Lab Tests Performed": updates.get("lab_tests", ""),
            "Radiology Tests": updates.get("radiology_tests", ""),
            "Date of Admission": updates.get("date_of_admission", datetime.now().strftime("%Y-%m-%d")),
            "Date of Discharge": updates.get("date_of_discharge", ""),
            "Follow-up Date": updates.get("follow_up_date", "")
        }
        
        # Re-write the Markdown reports
        with open(os.path.join(doc_dir, "patient_summary.md"), 'w', encoding='utf-8') as f:
            f.write(generate_patient_summary(gen_data))
        with open(os.path.join(doc_dir, "admission_report.md"), 'w', encoding='utf-8') as f:
            f.write(generate_admission_report(gen_data))
        with open(os.path.join(doc_dir, "doctor_notes.md"), 'w', encoding='utf-8') as f:
            f.write(generate_doctor_notes(gen_data))
        with open(os.path.join(doc_dir, "prescription.md"), 'w', encoding='utf-8') as f:
            f.write(generate_prescription(gen_data))
        with open(os.path.join(doc_dir, "lab_report.md"), 'w', encoding='utf-8') as f:
            f.write(generate_lab_report(gen_data))
        with open(os.path.join(doc_dir, "radiology_report.md"), 'w', encoding='utf-8') as f:
            f.write(generate_radiology_report(gen_data))
            
        status = gen_data["Current Status"]
        if status.strip().lower() == "discharged" and gen_data["Date of Discharge"]:
            with open(os.path.join(doc_dir, "discharge_summary.md"), 'w', encoding='utf-8') as f:
                f.write(generate_discharge_summary(gen_data))
                
        return f"Success: Patient {patient_id} record has been updated and clinical documents synchronized."
    except Exception as e:
        return f"Error updating patient record: {str(e)}"
