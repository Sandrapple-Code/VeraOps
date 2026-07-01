import os
import sys
import json
from datetime import datetime
from langchain_core.tools import tool
from db.sqlite import add_patient, get_patient

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
def register_patient(patient_data_json: str) -> str:
    """
    Registers a new patient in the hospital system.
    Inserts their record into the SQLite database, creates their document directory,
    and generates initial markdown clinical reports.
    
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
        
    patient_id = data.get("patient_id") or data.get("Patient ID")
    name = data.get("name") or data.get("Full Name")
    
    if not patient_id or not name:
        return "Error: patient_id and name are required fields."
        
    patient_id = str(patient_id).strip()
    name = str(name).strip()
    
    try:
        # 1. Check if patient already exists
        if get_patient(patient_id) is not None:
            return f"Error: A patient with ID {patient_id} already exists."
            
        # 2. Insert into SQLite
        db_data = {
            "patient_id": patient_id,
            "name": name,
            "age": int(data.get("age", data.get("Age", 0))),
            "gender": data.get("gender", data.get("Gender", "N/A")),
            "diagnosis": data.get("diagnosis", data.get("Disease / Diagnosis", "")),
            "medicines": data.get("medicines", data.get("Prescription", "")),
            "ward": data.get("ward", data.get("Ward", "")),
            "bed_number": data.get("bed_number", data.get("Bed Number", "")),
            "visit_notes": data.get("visit_notes", data.get("Doctor Notes", ""))
        }
        add_patient(db_data)
        
        # 3. Create document folder
        doc_dir = os.path.join(PROJECT_ROOT, "patient_documents", patient_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # 4. Map DB keys to generator structure
        gen_data = {
            "Patient ID": patient_id,
            "Full Name": name,
            "Age": db_data["age"],
            "Gender": db_data["gender"],
            "Blood Group": data.get("blood_group", data.get("Blood Group", "Unknown")),
            "Phone Number": data.get("phone_number", data.get("Phone Number", "N/A")),
            "Email": data.get("email", data.get("Email", "N/A")),
            "Address": data.get("address", data.get("Address", "N/A")),
            "Emergency Contact Name": data.get("emergency_contact_name", data.get("Emergency Contact Name", "N/A")),
            "Emergency Contact Number": data.get("emergency_contact_number", data.get("Emergency Contact Number", "N/A")),
            "Department": data.get("department", data.get("Department", "General Medicine")),
            "Assigned Doctor": data.get("assigned_doctor", data.get("Assigned Doctor", "Staff Physician")),
            "Ward": db_data["ward"],
            "Bed Number": db_data["bed_number"],
            "Disease / Diagnosis": db_data["diagnosis"],
            "Symptoms": data.get("symptoms", data.get("Symptoms", db_data["diagnosis"])),
            "Past Medical History": data.get("past_medical_history", data.get("Past Medical History", "None")),
            "Allergies": data.get("allergies", data.get("Allergies", "NKDA")),
            "Current Medications": data.get("current_medications", data.get("Current Medications", "None")),
            "Treatment Plan": data.get("treatment_plan", data.get("Treatment Plan", "Standard supportive care.")),
            "Current Status": data.get("current_status", data.get("Current Status", "Admitted")),
            "Admission Summary": data.get("admission_summary", data.get("Admission Summary", "Patient admitted.")),
            "Doctor Notes": db_data["visit_notes"],
            "Prescription": db_data["medicines"],
            "Lab Tests Performed": data.get("lab_tests", data.get("Lab Tests Performed", "")),
            "Radiology Tests": data.get("radiology_tests", data.get("Radiology Tests", "")),
            "Date of Admission": data.get("date_of_admission", data.get("Date of Admission", datetime.now().strftime("%Y-%m-%d"))),
            "Date of Discharge": data.get("date_of_discharge", data.get("Date of Discharge", "")),
            "Follow-up Date": data.get("follow_up_date", data.get("Follow-up Date", ""))
        }
        
        # Write markdown reports
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
                
        return f"Success: Patient {name} (ID: {patient_id}) has been registered and clinical files created."
    except Exception as e:
        return f"Error during patient registration: {str(e)}"
