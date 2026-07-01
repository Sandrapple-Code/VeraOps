import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the project root to sys.path so we can import from db.sqlite
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from db.sqlite import add_patient, get_patient

# Define document directory path
DOCS_DIR = os.path.join(PROJECT_ROOT, "patient_documents")

def load_patients_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Loads patient data from the specified JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Patients JSON file not found at: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def insert_patient_to_db(patient: Dict[str, Any]) -> bool:
    """
    Inserts a patient record into the SQLite database if not already present.
    Returns True if inserted, False if skipped.
    """
    patient_id = patient.get("Patient ID")
    if not patient_id:
        print("[ERROR] Skip record: Missing 'Patient ID'")
        return False

    # Check if patient already exists to prevent duplicate insertion
    if get_patient(patient_id) is not None:
        return False

    # Map patients.json fields to SQLite database fields
    db_data = {
        "patient_id": patient_id,
        "name": patient.get("Full Name"),
        "age": int(patient.get("Age", 0)),
        "gender": patient.get("Gender"),
        "diagnosis": patient.get("Disease / Diagnosis"),
        "medicines": patient.get("Prescription"),
        "ward": patient.get("Ward"),
        "bed_number": patient.get("Bed Number"),
        "visit_notes": patient.get("Doctor Notes")
    }

    try:
        add_patient(db_data)
        return True
    except Exception as e:
        print(f"[ERROR] Error inserting patient {patient_id} into database: {e}")
        return False

def calculate_stay(adm_date: str, dis_date: str) -> str:
    """
    Calculates the duration of hospital stay in days.
    """
    if not adm_date or not dis_date:
        return "N/A"
    try:
        a = datetime.strptime(adm_date.strip(), "%Y-%m-%d")
        d = datetime.strptime(dis_date.strip(), "%Y-%m-%d")
        days = (d - a).days
        return f"{days} days"
    except Exception:
        return "N/A"

def parse_prescription(prescription_str: str) -> List[Dict[str, str]]:
    """
    Parses a combined prescription string into structured components.
    """
    parsed_medicines = []
    if not prescription_str:
        return parsed_medicines
        
    items = [item.strip() for item in prescription_str.split(";") if item.strip()]
    for item in items:
        parts = item.split()
        if not parts:
            continue
            
        name = parts[0]
        dosage = "N/A"
        frequency = "N/A"
        route_instructions = []
        
        # Detect dosage by looking for metric suffix
        for p in parts[1:]:
            if any(unit in p.lower() for unit in ["mg", "mcg", "meq", "g", "ml", "tab", "cap"]):
                dosage = p
                break
        
        # Detect route & frequency
        for p in parts[1:]:
            p_lower = p.lower()
            if p_lower in ["qd", "bid", "tid", "qid", "prn", "q8h", "q12h", "q6h"]:
                frequency = p.upper()
            elif p_lower in ["po", "iv", "sc", "im", "subcutaneous"]:
                route_instructions.append(p.upper())
                
        if "(" in item and ")" in item:
            start_idx = item.find("(")
            end_idx = item.find(")")
            route_instructions.append(item[start_idx+1 : end_idx])
            
        instr_str = ", ".join(route_instructions) if route_instructions else "PO"
        parsed_medicines.append({
            "name": name,
            "dosage": dosage if dosage != "N/A" else (parts[1] if len(parts) > 1 else "N/A"),
            "frequency": frequency if frequency != "N/A" else "Scheduled",
            "instructions": instr_str
        })
    return parsed_medicines

def parse_lab_tests(lab_str: str) -> List[Dict[str, str]]:
    """
    Parses a lab test result summary into a list of structured records.
    """
    results = []
    if not lab_str:
        return results
        
    # Split by comma but respect internal parentheses
    raw_parts = lab_str.split(",")
    parts = []
    current_part = ""
    for p in raw_parts:
        if current_part:
            current_part += "," + p
        else:
            current_part = p
        if current_part.count("(") == current_part.count(")"):
            parts.append(current_part.strip())
            current_part = ""
            
    for part in parts:
        if not part:
            continue
        if "(" in part and ")" in part:
            test_name = part[:part.find("(")].strip()
            result_info = part[part.find("(")+1 : part.find(")")].strip()
            remark = "Normal"
            result_lower = result_info.lower()
            if any(w in result_lower for w in ["elevated", "high", "abnormal", "positive", "low", "severe", "hyper"]):
                remark = "Abnormal"
            elif "negative" in result_lower:
                remark = "Normal"
        else:
            test_name = part
            result_info = "Completed"
            remark = "Normal"
            
        results.append({
            "test": test_name,
            "result": result_info,
            "remark": remark
        })
    return results

def generate_patient_summary(patient: Dict[str, Any]) -> str:
    return f"""# Patient Summary - {patient.get('Patient ID', 'N/A')}

## Demographic Information
- **Patient ID:** {patient.get('Patient ID', 'N/A')}
- **Name:** {patient.get('Full Name', 'N/A')}
- **Age:** {patient.get('Age', 'N/A')}
- **Gender:** {patient.get('Gender', 'N/A')}
- **Blood Group:** {patient.get('Blood Group', 'N/A')}

## Contact Details
- **Phone Number:** {patient.get('Phone Number', 'N/A')}
- **Email:** {patient.get('Email', 'N/A')}
- **Address:** {patient.get('Address', 'N/A')}
- **Emergency Contact:** {patient.get('Emergency Contact Name', 'N/A')} ({patient.get('Emergency Contact Number', 'N/A')})

## Administrative Info
- **Department:** {patient.get('Department', 'N/A')}
- **Assigned Doctor:** {patient.get('Assigned Doctor', 'N/A')}
- **Ward:** {patient.get('Ward', 'N/A')}
- **Bed Number:** {patient.get('Bed Number', 'N/A')}
- **Admission Type:** {patient.get('Admission Type', 'N/A')}
- **Current Status:** {patient.get('Current Status', 'N/A')}

## Clinical Information
- **Chief Complaint:** {patient.get('Chief Complaint', 'N/A')}
- **Primary Diagnosis:** {patient.get('Disease / Diagnosis', 'N/A')}
- **Symptoms:** {patient.get('Symptoms', 'N/A')}
- **Past Medical History:** {patient.get('Past Medical History', 'N/A')}
- **Known Allergies:** {patient.get('Allergies', 'N/A')}
- **Current Medications:** {patient.get('Current Medications', 'N/A')}

## Measurements & Billing
- **Height:** {patient.get('Height', 'N/A')} cm
- **Weight:** {patient.get('Weight', 'N/A')} kg
- **BMI:** {patient.get('BMI', 'N/A')}
- **Insurance Provider:** {patient.get('Insurance Provider', 'N/A')}
- **Insurance Number:** {patient.get('Insurance Number', 'N/A')}
- **National ID / Hospital ID:** {patient.get('National ID', 'N/A')}
"""

def generate_admission_report(patient: Dict[str, Any]) -> str:
    return f"""# Admission Report

- **Patient ID:** {patient.get('Patient ID', 'N/A')}
- **Patient Name:** {patient.get('Full Name', 'N/A')}
- **Date of Admission:** {patient.get('Date of Admission', 'N/A')}
- **Admitting Doctor:** {patient.get('Assigned Doctor', 'N/A')}
- **Department:** {patient.get('Department', 'N/A')}
- **Ward Allocation:** {patient.get('Ward', 'N/A')} (Bed: {patient.get('Bed Number', 'N/A')})
- **Admission Type:** {patient.get('Admission Type', 'N/A')}
- **Current Status:** {patient.get('Current Status', 'N/A')}

## Chief Complaint & Admission History
{patient.get('Chief Complaint', 'No chief complaint recorded.')}

## Initial Clinical Presentation
- **Symptoms:** {patient.get('Symptoms', 'N/A')}
- **Primary Diagnosis:** {patient.get('Disease / Diagnosis', 'N/A')}
- **Known Allergies:** {patient.get('Allergies', 'NKDA')}

## Vital Statistics
- **Height:** {patient.get('Height', 'N/A')} cm
- **Weight:** {patient.get('Weight', 'N/A')} kg
- **BMI:** {patient.get('BMI', 'N/A')}
"""

def generate_doctor_notes(patient: Dict[str, Any]) -> str:
    return f"""# Doctor Notes

## Admission Diagnosis
{patient.get('Disease / Diagnosis', 'N/A')}

## Chief Complaint
{patient.get('Chief Complaint', 'N/A')}

## Symptoms
{patient.get('Symptoms', 'N/A')}

## Initial Assessment
Patient presented with chief complaints of: {patient.get('Chief Complaint', 'N/A')}.
Vitals are stable on admission. Mentally alert and oriented x3.

## Clinical Impression
Admitted for further management of {patient.get('Disease / Diagnosis', 'N/A')}. Patient will be monitored closely in the {patient.get('Ward', 'ward')}.
"""

def generate_prescription(patient: Dict[str, Any]) -> str:
    return f"""# Medical Prescription

## Known Allergies
{patient.get('Allergies', 'NKDA')}

## Current Medications
{patient.get('Current Medications', 'None')}

## Medication Notes
Adhere strictly to the recommended treatment. If any adverse symptoms occur, notify the attending physician immediately.
"""

def generate_lab_report(patient: Dict[str, Any]) -> str:
    return f"""# Laboratory Test Report

## Pending Laboratory Investigations
- CBC (Complete Blood Count)
- BMP (Basic Metabolic Panel)
- Liver Function Tests (LFT)
- Serum Electrolytes
- Coagulation Profile
"""

def generate_radiology_report(patient: Dict[str, Any]) -> str:
    return f"""# Radiology & Imaging Report

## Pending Radiology Investigations
- Chest X-ray (CXR)
- Focused assessment / scan based on diagnosis: {patient.get('Disease / Diagnosis', 'N/A')}
"""

def generate_treatment_plan(patient: Dict[str, Any]) -> str:
    return f"""# Initial Treatment Plan

## Initial Treatment Plan
1. Telemetry / Bedside vitals monitoring every 4 hours.
2. Maintain IV access; start supportive hydration if clinically indicated.
3. Manage symptoms as per clinical pathway for {patient.get('Disease / Diagnosis', 'N/A')}.

## Monitoring Plan
- Continuous SpO2 and Blood Pressure monitoring.
- Track intake/output hourly.

## Follow-up
- Daily physician rounds.
- Re-evaluate for potential discharge parameters on clinical stabilization.
"""

def generate_discharge_summary(patient: Dict[str, Any]) -> str:
    adm_date = patient.get("Date of Admission", "")
    dis_date = patient.get("Date of Discharge", "")
    stay = calculate_stay(adm_date, dis_date)
    
    return f"""# Discharge Summary

- **Patient ID:** {patient.get('Patient ID', 'N/A')}
- **Patient Name:** {patient.get('Full Name', 'N/A')}
- **Date of Admission:** {adm_date}
- **Date of Discharge:** {dis_date}
- **Hospital Stay Duration:** {stay}
- **Attending Physician:** {patient.get('Assigned Doctor', 'N/A')}

## Discharge Diagnosis
{patient.get('Disease / Diagnosis', 'N/A')}

## Summary of Hospital Course & Treatment
{patient.get('Discharge Summary', 'Patient treated as per protocol and discharged in stable condition.')}

## Condition at Discharge
- **Vitals:** Stable
- **Mental Status:** Alert, oriented
- **Mobility:** Ambulated safely on discharge

## Discharge Medications
- **Prescription:** {patient.get('Prescription', 'No outpatient medicines prescribed.')}

## Follow-Up Instructions
- **Follow-up Appointment:** Scheduled for {patient.get('Follow-up Date', 'N/A')} with {patient.get('Assigned Doctor', 'N/A')}.
- Follow low-sodium, diabetic, or therapeutic diet as instructed.
- Seek immediate medical attention if symptoms recur.
"""

def seed_database(json_path: str):
    print("[START] Starting database and document seeding process...")
    
    patients = load_patients_json(json_path)
    print(f"Total patient records found in JSON: {len(patients)}")
    
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    imported_count = 0
    skipped_count = 0
    
    for idx, patient in enumerate(patients):
        patient_id = patient.get("Patient ID")
        name = patient.get("Full Name", "Unknown")
        
        # 1. Insert into Database
        inserted = insert_patient_to_db(patient)
        if inserted:
            print(f"\n[OK] Imported Patient {patient_id} ({name}) to database")
            imported_count += 1
        else:
            print(f"\n[SKIP] Skipping duplicate patient {patient_id} database record")
            skipped_count += 1
            
        # 2. Folder Creation
        patient_dir = os.path.join(DOCS_DIR, patient_id)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
            print(f"  [OK] Created Folder {patient_id}/")
        else:
            print(f"  [SKIP] Folder {patient_id}/ already exists")
            
        # 3. Document Generation
        # Summary
        summary_path = os.path.join(patient_dir, "patient_summary.md")
        if not os.path.exists(summary_path):
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(generate_patient_summary(patient))
            print("  [OK] Generated Patient Summary")
            
        # Admission Report
        adm_path = os.path.join(patient_dir, "admission_report.md")
        if not os.path.exists(adm_path):
            with open(adm_path, 'w', encoding='utf-8') as f:
                f.write(generate_admission_report(patient))
            print("  [OK] Generated Admission Report")
            
        # Doctor Notes
        notes_path = os.path.join(patient_dir, "doctor_notes.md")
        if not os.path.exists(notes_path):
            with open(notes_path, 'w', encoding='utf-8') as f:
                f.write(generate_doctor_notes(patient))
            print("  [OK] Generated Doctor Notes")
            
        # Prescription
        rx_path = os.path.join(patient_dir, "prescription.md")
        if not os.path.exists(rx_path):
            with open(rx_path, 'w', encoding='utf-8') as f:
                f.write(generate_prescription(patient))
            print("  [OK] Generated Prescription")
            
        # Lab Report
        lab_path = os.path.join(patient_dir, "lab_report.md")
        if not os.path.exists(lab_path):
            with open(lab_path, 'w', encoding='utf-8') as f:
                f.write(generate_lab_report(patient))
            print("  [OK] Generated Lab Report")
            
        # Radiology Report
        rad_path = os.path.join(patient_dir, "radiology_report.md")
        if not os.path.exists(rad_path):
            with open(rad_path, 'w', encoding='utf-8') as f:
                f.write(generate_radiology_report(patient))
            print("  [OK] Generated Radiology Report")
            
        # Discharge Summary (only if status is "Discharged")
        status = patient.get("Current Status", "")
        if status.strip().lower() == "discharged":
            discharge_path = os.path.join(patient_dir, "discharge_summary.md")
            if not os.path.exists(discharge_path):
                with open(discharge_path, 'w', encoding='utf-8') as f:
                    f.write(generate_discharge_summary(patient))
                print("  [OK] Generated Discharge Summary")
                
    print("\n====================================================")
    print("Seeding completed successfully!")
    print(f"SQLite DB: {imported_count} imported, {skipped_count} skipped (duplicates)")
    print(f"Total patient directories managed: {len(patients)}")
    print("====================================================")

if __name__ == "__main__":
    json_file = os.path.join(PROJECT_ROOT, "data", "patients.json")
    seed_database(json_file)
