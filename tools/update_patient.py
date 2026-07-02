import os
import sys
import json
from datetime import datetime
from langchain_core.tools import tool
from db.sqlite import update_patient, get_patient, occupy_bed, release_bed, add_timeline_event
from rag.ingestion import index_patient_documents

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
    generate_treatment_plan,
    generate_discharge_summary
)

@tool
def modify_patient_record(patient_id: str, updates_json: str) -> str:
    """
    Updates an existing patient's clinical and administrative details in the SQLite database
    and synchronizes their local Markdown clinical documents.
    
    :param patient_id: Unique Patient ID (e.g., 'P001').
    :param updates_json: JSON string containing fields to update (e.g. {"ward": "ICU", "bed_number": "ICU-002"}).
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
            
        old_ward = record.get("ward")
        old_bed = record.get("bed_number")
        
        # 2. Update SQLite
        db_success = update_patient(patient_id, updates)
        
        # 3. Retrieve the fully updated patient record
        updated_record = get_patient(patient_id)
        
        # 4. Handle Bed occupancy shifts if ward or bed changed
        new_ward = updated_record.get("ward")
        new_bed = updated_record.get("bed_number")
        if (new_ward != old_ward) or (new_bed != old_bed):
            if new_ward and new_bed:
                occupy_bed(new_ward, new_bed, patient_id, updated_record.get("date_of_admission") or datetime.now().strftime("%Y-%m-%d"))
                add_timeline_event(patient_id, "Bed Allocation", f"Transferred to ward {new_ward}, bed {new_bed} via AI Agent.", updated_record.get("assigned_doctor"))
            else:
                release_bed(patient_id)
                add_timeline_event(patient_id, "Discharge", f"Released bed allocation via AI Agent.", updated_record.get("assigned_doctor"))
                
        # Log to timeline
        changed_fields = list(updates.keys())
        if changed_fields:
            desc = f"Updated clinical/admin fields via AI Agent: {', '.join(changed_fields)}."
            if "visit_notes" in updates:
                add_timeline_event(patient_id, "Doctor Notes Added", "Appended clinical progress notes via AI Agent.", updated_record.get("assigned_doctor"))
            elif "medicines" in updates:
                add_timeline_event(patient_id, "Prescription Updated", "Prescription modified via AI Agent.", updated_record.get("assigned_doctor"))
            else:
                add_timeline_event(patient_id, "Patient Record Updated", desc, updated_record.get("assigned_doctor"))
                
        # 5. Regenerate Markdown clinical files to keep them synchronized
        doc_dir = os.path.join(PROJECT_ROOT, "patient_documents", patient_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Merge existing updates to rebuild JSON format for template generators
        gen_data = {
            "Patient ID": patient_id,
            "Full Name": updated_record.get("name"),
            "Age": updated_record.get("age"),
            "Gender": updated_record.get("gender"),
            "Blood Group": updated_record.get("blood_group") or updates.get("blood_group", "Unknown"),
            "Phone Number": updated_record.get("phone_number") or updates.get("phone_number", "N/A"),
            "Email": updated_record.get("email") or updates.get("email", "N/A"),
            "Address": updated_record.get("address") or updates.get("address", "N/A"),
            "Emergency Contact Name": updated_record.get("emergency_contact_name") or updates.get("emergency_contact_name", "N/A"),
            "Emergency Contact Number": updated_record.get("emergency_contact_number") or updates.get("emergency_contact_number", "N/A"),
            "Department": updated_record.get("department") or updates.get("department", "General Medicine"),
            "Assigned Doctor": updated_record.get("assigned_doctor") or updates.get("assigned_doctor", "Staff Physician"),
            "Ward": updated_record.get("ward"),
            "Bed Number": updated_record.get("bed_number"),
            "Admission Type": updated_record.get("admission_type") or updates.get("admission_type", "Emergency"),
            "Current Status": updated_record.get("current_status") or updates.get("current_status", "Admitted"),
            "Chief Complaint": updated_record.get("chief_complaint") or updates.get("chief_complaint", ""),
            "Disease / Diagnosis": updated_record.get("diagnosis"),
            "Symptoms": updated_record.get("symptoms") or updates.get("symptoms", updated_record.get("diagnosis")),
            "Past Medical History": updated_record.get("past_medical_history") or updates.get("past_medical_history", "None"),
            "Allergies": updated_record.get("allergies") or updates.get("allergies", "NKDA"),
            "Current Medications": updated_record.get("current_medications") or updates.get("current_medications", "None"),
            "Height": updated_record.get("height") if updated_record.get("height") is not None else "N/A",
            "Weight": updated_record.get("weight") if updated_record.get("weight") is not None else "N/A",
            "BMI": updated_record.get("bmi") if updated_record.get("bmi") is not None else "N/A",
            "Insurance Provider": updated_record.get("insurance_provider") or updates.get("insurance_provider", "N/A"),
            "Insurance Number": updated_record.get("insurance_number") or updates.get("insurance_number", "N/A"),
            "National ID": updated_record.get("national_id") or updates.get("national_id", "N/A"),
            "Date of Admission": updated_record.get("date_of_admission") or updated_record.get("date_of_admission", datetime.now().strftime("%Y-%m-%d")),
            "Date of Discharge": updated_record.get("date_of_discharge") or updated_record.get("date_of_discharge", ""),
            "Follow-up Date": updated_record.get("follow_up_date") or updates.get("follow_up_date", ""),
            "Prescription": updated_record.get("medicines"),
            "Doctor Notes": updated_record.get("visit_notes")
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
        with open(os.path.join(doc_dir, "treatment_plan.md"), 'w', encoding='utf-8') as f:
            f.write(generate_treatment_plan(gen_data))
            
        status = gen_data["Current Status"]
        if status.strip().lower() == "discharged" and gen_data["Date of Discharge"]:
            with open(os.path.join(doc_dir, "discharge_summary.md"), 'w', encoding='utf-8') as f:
                f.write(generate_discharge_summary(gen_data))
                
        # Re-index the files for this patient
        index_patient_documents(patient_id)
        
        return f"Success: Patient {patient_id} record has been updated and clinical documents synchronized."
    except Exception as e:
        return f"Error updating patient record: {str(e)}"
