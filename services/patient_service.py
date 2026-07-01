import os
import sys
from datetime import datetime
from typing import Dict, Any

# Project Root Helper
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from db.sqlite import add_patient, get_patient, occupy_bed, release_bed
from rag.ingestion import index_patient_documents
from scripts.seed_database import (
    generate_patient_summary,
    generate_admission_report,
    generate_doctor_notes,
    generate_prescription,
    generate_lab_report,
    generate_radiology_report,
    generate_treatment_plan,
    generate_discharge_summary
)

def register_new_patient(data: Dict[str, Any]) -> str:
    """
    Unified service to register a new patient:
    1. Saves patient record to SQLite.
    2. Reserves selected hospital bed.
    3. Generates Markdown documents under patient_documents/PXXX/.
    4. Dynamically builds FAISS vector index for only this patient.
    
    :param data: Dictionary containing patient details.
    :return: A success message string.
    """
    patient_id = data.get("patient_id") or data.get("Patient ID")
    name = data.get("name") or data.get("Full Name")
    
    if not patient_id or not name:
        raise ValueError("patient_id and name are required fields.")
        
    patient_id = str(patient_id).strip()
    name = str(name).strip()
    
    # 1. Check if patient already exists
    if get_patient(patient_id) is not None:
        raise ValueError(f"A patient with ID {patient_id} already exists.")
        
    # 2. Extract SQLite data and map fields
    db_data = {
        "patient_id": patient_id,
        "name": name,
        "age": int(data.get("age", data.get("Age", 0))),
        "gender": data.get("gender", data.get("Gender", "N/A")),
        "blood_group": data.get("blood_group", data.get("Blood Group", "N/A")),
        "phone_number": data.get("phone_number", data.get("Phone Number", "N/A")),
        "email": data.get("email", data.get("Email", "N/A")),
        "address": data.get("address", data.get("Address", "N/A")),
        "emergency_contact_name": data.get("emergency_contact_name", data.get("Emergency Contact Name", "N/A")),
        "emergency_contact_number": data.get("emergency_contact_number", data.get("Emergency Contact Number", "N/A")),
        "date_of_admission": data.get("date_of_admission", datetime.now().strftime("%Y-%m-%d")),
        "department": data.get("department", data.get("Department", "General Medicine")),
        "assigned_doctor": data.get("assigned_doctor", data.get("Assigned Doctor", "Staff Physician")),
        "ward": data.get("ward", data.get("Ward", "")),
        "bed_number": data.get("bed_number", data.get("Bed Number", "")),
        "admission_type": data.get("admission_type", data.get("Admission Type", "Emergency")),
        "current_status": data.get("current_status", data.get("Current Status", "Admitted")),
        "chief_complaint": data.get("chief_complaint", data.get("Chief Complaint", "")),
        "diagnosis": data.get("diagnosis", data.get("Disease / Diagnosis", "")),
        "symptoms": data.get("symptoms", data.get("Symptoms", "")),
        "past_medical_history": data.get("past_medical_history", data.get("Past Medical History", "")),
        "allergies": data.get("allergies", data.get("Allergies", "NKDA")),
        "current_medications": data.get("current_medications", data.get("Current Medications", "None")),
        "height": data.get("height") or data.get("Height"),
        "weight": data.get("weight") or data.get("Weight"),
        "bmi": data.get("bmi") or data.get("BMI"),
        "insurance_provider": data.get("insurance_provider", data.get("Insurance Provider", "N/A")),
        "insurance_number": data.get("insurance_number", data.get("Insurance Number", "N/A")),
        "national_id": data.get("national_id", data.get("National ID", "N/A")),
        "date_of_discharge": data.get("date_of_discharge", ""),
        "discharge_summary": data.get("discharge_summary", ""),
        "follow_up_date": data.get("follow_up_date", ""),
        "medicines": data.get("medicines", data.get("Prescription", "")),
        "visit_notes": data.get("visit_notes", data.get("Doctor Notes", ""))
    }
    
    # Clean Real/Numeric columns
    for float_col in ["height", "weight", "bmi"]:
        val = db_data[float_col]
        if val is not None:
            try:
                db_data[float_col] = float(val)
            except Exception:
                db_data[float_col] = None
                
    # Insert to SQLite
    add_patient(db_data)
    
    # 3. Reserve Selected Bed
    if db_data["ward"] and db_data["bed_number"]:
        occupy_bed(db_data["ward"], db_data["bed_number"], patient_id, db_data["date_of_admission"])
        
    # 4. Create document folder and generate Markdown files
    doc_dir = os.path.join(PROJECT_ROOT, "patient_documents", patient_id)
    os.makedirs(doc_dir, exist_ok=True)
    
    # Prepare standard mapping dictionary for generator templates
    gen_data = {
        "Patient ID": patient_id,
        "Full Name": name,
        "Age": db_data["age"],
        "Gender": db_data["gender"],
        "Blood Group": db_data["blood_group"],
        "Phone Number": db_data["phone_number"],
        "Email": db_data["email"],
        "Address": db_data["address"],
        "Emergency Contact Name": db_data["emergency_contact_name"],
        "Emergency Contact Number": db_data["emergency_contact_number"],
        "Department": db_data["department"],
        "Assigned Doctor": db_data["assigned_doctor"],
        "Ward": db_data["ward"],
        "Bed Number": db_data["bed_number"],
        "Admission Type": db_data["admission_type"],
        "Current Status": db_data["current_status"],
        "Chief Complaint": db_data["chief_complaint"],
        "Disease / Diagnosis": db_data["diagnosis"],
        "Symptoms": db_data["symptoms"],
        "Past Medical History": db_data["past_medical_history"],
        "Allergies": db_data["allergies"],
        "Current Medications": db_data["current_medications"],
        "Height": db_data["height"] if db_data["height"] is not None else "N/A",
        "Weight": db_data["weight"] if db_data["weight"] is not None else "N/A",
        "BMI": db_data["bmi"] if db_data["bmi"] is not None else "N/A",
        "Insurance Provider": db_data["insurance_provider"],
        "Insurance Number": db_data["insurance_number"],
        "National ID": db_data["national_id"],
        "Date of Admission": db_data["date_of_admission"],
        "Date of Discharge": db_data["date_of_discharge"],
        "Follow-up Date": db_data["follow_up_date"],
        "Prescription": db_data["medicines"],
        "Doctor Notes": db_data["visit_notes"]
    }
    
    # Write Markdown files
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
    with open(os.path.join(doc_dir, "treatment_plan.md"), 'w', encoding='utf-8') as f:
        f.write(generate_treatment_plan(gen_data))
        
    status = gen_data["Current Status"]
    if status.strip().lower() == "discharged" and gen_data["Date of Discharge"]:
        with open(os.path.join(doc_dir, "discharge_summary.md"), 'w', encoding='utf-8') as f:
            f.write(generate_discharge_summary(gen_data))
            
    # 5. Generate Embedding and add to FAISS patient store incrementally
    index_patient_documents(patient_id)
    
    return f"Success: Patient {name} (ID: {patient_id}) has been registered, clinical files created, and FAISS index updated."
