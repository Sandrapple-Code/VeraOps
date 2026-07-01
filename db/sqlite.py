import sqlite3
import os
from typing import Dict, Any, List, Optional

# Define the database path to be in the project root directory
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hospital.db")

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Creates and returns a SQLite connection to the hospital database.
    Ensures that the directory for the database exists.
    
    :param db_path: Path to the SQLite database file.
    :return: A sqlite3.Connection object with row_factory set to sqlite3.Row.
    """
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    # Enable Row factory to access columns by name as a dictionary-like object
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_PATH) -> None:
    """
    Initializes the database schema, handles table migrations, and populates the beds table.
    """
    conn = get_connection(db_path)
    try:
        # Check if old patients table needs migration
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
        table_exists = cursor.fetchone()
        
        needs_migration = False
        if table_exists:
            cursor = conn.execute("PRAGMA table_info(patients)")
            cols = [col[1] for col in cursor.fetchall()]
            # If it lacks newer fields (like blood_group), flag for migration
            if "blood_group" not in cols:
                needs_migration = True

        create_patients_sql = """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT,
            blood_group TEXT,
            phone_number TEXT,
            email TEXT,
            address TEXT,
            emergency_contact_name TEXT,
            emergency_contact_number TEXT,
            date_of_admission TEXT,
            department TEXT,
            assigned_doctor TEXT,
            ward TEXT,
            bed_number TEXT,
            admission_type TEXT,
            current_status TEXT,
            chief_complaint TEXT,
            diagnosis TEXT,
            symptoms TEXT,
            past_medical_history TEXT,
            allergies TEXT,
            current_medications TEXT,
            height REAL,
            weight REAL,
            bmi REAL,
            insurance_provider TEXT,
            insurance_number TEXT,
            national_id TEXT,
            date_of_discharge TEXT,
            discharge_summary TEXT,
            follow_up_date TEXT,
            medicines TEXT,
            visit_notes TEXT
        );
        """

        create_beds_sql = """
        CREATE TABLE IF NOT EXISTS beds (
            bed_number TEXT,
            ward TEXT,
            status TEXT DEFAULT 'Available',
            occupied_by TEXT,
            admission_date TEXT,
            PRIMARY KEY (ward, bed_number)
        );
        """

        if needs_migration:
            print("[MIGRATION] Migrating patients table to new schema...")
            conn.execute("ALTER TABLE patients RENAME TO patients_old")
            conn.execute(create_patients_sql)
            
            # Map old columns to new columns, fill in defaults for required new columns
            conn.execute("""
            INSERT INTO patients (
                patient_id, name, age, gender, diagnosis, medicines, ward, bed_number, visit_notes,
                date_of_admission, department, assigned_doctor, current_status, chief_complaint, symptoms, past_medical_history, allergies, current_medications
            )
            SELECT 
                patient_id, name, age, gender, diagnosis, medicines, ward, bed_number, visit_notes,
                '2026-06-15', -- default admission date
                'General Medicine', -- default department
                'Staff Physician', -- default doctor
                'Admitted', -- default status
                diagnosis, -- chief complaint (fallback to diagnosis)
                diagnosis, -- symptoms (fallback)
                'None', -- past medical history (fallback)
                'NKDA', -- allergies (fallback)
                'None' -- current medications (fallback)
            FROM patients_old
            """)
            conn.execute("DROP TABLE patients_old")
            conn.commit()
            print("[MIGRATION] patients table successfully migrated.")
        else:
            conn.execute(create_patients_sql)
            conn.commit()

        # Create beds table
        conn.execute(create_beds_sql)
        conn.commit()

        # Check if beds table is empty and populate it
        cursor = conn.execute("SELECT COUNT(*) FROM beds")
        beds_count = cursor.fetchone()[0]
        if beds_count == 0:
            print("[MIGRATION] Populating beds table with 10 wards x 100 beds...")
            wards = [
                ("Cardiology", "CARD"),
                ("Neurology", "NEUR"),
                ("Orthopedics", "ORTH"),
                ("Emergency", "ER"),
                ("ICU", "ICU"),
                ("General Medicine", "GEN"),
                ("Pulmonology", "PULM"),
                ("Nephrology", "NEPH"),
                ("Pediatrics", "PED"),
                ("Surgery", "SURG")
            ]
            for ward, prefix in wards:
                for i in range(1, 101):
                    bed_num = f"{prefix}-{i:03d}"
                    conn.execute(
                        "INSERT INTO beds (bed_number, ward, status, occupied_by, admission_date) VALUES (?, ?, 'Available', NULL, NULL)",
                        (bed_num, ward)
                    )
            conn.commit()
            print("[MIGRATION] Populated 1000 beds.")

            # Assign active patients from the database to valid beds in new wards
            cursor = conn.execute("SELECT * FROM patients WHERE ward IS NOT NULL AND ward != '' AND current_status != 'Discharged'")
            active_patients = cursor.fetchall()
            
            ward_mapping = {
                "Cardiovascular Sub-Acute Unit": "Cardiology",
                "Surgical Ward B": "Surgery",
                "Stroke Unit": "Neurology",
                "General Ward A": "General Medicine",
                "ICU": "ICU"
            }
            
            for p in active_patients:
                p_id = p["patient_id"]
                legacy_ward = p["ward"]
                new_ward = ward_mapping.get(legacy_ward, "General Medicine")
                adm_date = p["date_of_admission"] or "2026-06-15"
                
                # Find first available bed in new_ward
                cursor = conn.execute(
                    "SELECT bed_number FROM beds WHERE ward = ? AND status = 'Available' ORDER BY bed_number ASC LIMIT 1",
                    (new_ward,)
                )
                bed_row = cursor.fetchone()
                if bed_row:
                    new_bed = bed_row["bed_number"]
                    # Mark bed as occupied
                    conn.execute(
                        "UPDATE beds SET status = 'Occupied', occupied_by = ?, admission_date = ? WHERE ward = ? AND bed_number = ?",
                        (p_id, adm_date, new_ward, new_bed)
                    )
                    # Update patient ward and bed in patients table
                    conn.execute(
                        "UPDATE patients SET ward = ?, bed_number = ?, date_of_admission = ? WHERE patient_id = ?",
                        (new_ward, new_bed, adm_date, p_id)
                    )
            conn.commit()
            print("[MIGRATION] Migrated active patients to new beds.")
    finally:
        conn.close()

# Automatically initialize database when the module is imported
init_db()

def add_patient(data: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """
    Adds a new patient to the database dynamically based on keys present in data.
    """
    if "patient_id" not in data:
        raise ValueError("Missing 'patient_id' in patient data")
        
    # Get columns of patients table to filter keys
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("PRAGMA table_info(patients)")
        columns = [col[1] for col in cursor.fetchall()]
    finally:
        conn.close()
        
    # Filter data keys to match database columns
    insert_data = {k: v for k, v in data.items() if k in columns}
    if not insert_data:
        return
        
    fields = list(insert_data.keys())
    values = list(insert_data.values())
    
    placeholders = ", ".join(["?"] * len(fields))
    cols = ", ".join(fields)
    sql = f"INSERT INTO patients ({cols}) VALUES ({placeholders})"
    
    conn = get_connection(db_path)
    try:
        conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()

def get_patient(patient_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single patient by their patient_id.
    """
    sql = "SELECT * FROM patients WHERE patient_id = ?"
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(sql, (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def update_patient(patient_id: str, data: Dict[str, Any], db_path: str = DB_PATH) -> bool:
    """
    Updates an existing patient's details dynamically based on provided fields in data.
    """
    # Get columns of patients table to filter keys
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("PRAGMA table_info(patients)")
        columns = [col[1] for col in cursor.fetchall()]
    finally:
        conn.close()

    # Exclude patient_id from the update fields to avoid modifying the primary key
    update_data = {k: v for k, v in data.items() if k != "patient_id" and k in columns}
    if not update_data:
        return False
        
    fields = list(update_data.keys())
    values = list(update_data.values())
    
    # Construct dynamic SET clause
    set_clause = ", ".join([f"{field} = ?" for field in fields])
    sql = f"UPDATE patients SET {set_clause} WHERE patient_id = ?"
    
    # Add patient_id as the parameter for the WHERE clause
    values.append(patient_id)
    
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(sql, values)
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def get_all_patients(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Retrieves all patients from the database.
    """
    sql = "SELECT * FROM patients"
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_available_beds(ward: str, db_path: str = DB_PATH) -> List[str]:
    """
    Retrieves all unallocated bed numbers for a specific ward.
    """
    sql = "SELECT bed_number FROM beds WHERE ward = ? AND status = 'Available' ORDER BY bed_number ASC"
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(sql, (ward,))
        rows = cursor.fetchall()
        return [row["bed_number"] for row in rows]
    finally:
        conn.close()

def occupy_bed(ward: str, bed_number: str, patient_id: str, admission_date: str, db_path: str = DB_PATH) -> None:
    """
    Marks a specific bed in a ward as Occupied by the patient. Releases any other beds occupied by the patient.
    """
    release_bed(patient_id, db_path)
    sql = "UPDATE beds SET status = 'Occupied', occupied_by = ?, admission_date = ? WHERE ward = ? AND bed_number = ?"
    conn = get_connection(db_path)
    try:
        conn.execute(sql, (patient_id, admission_date, ward, bed_number))
        conn.commit()
    finally:
        conn.close()

def release_bed(patient_id: str, db_path: str = DB_PATH) -> None:
    """
    Sets the bed previously occupied by this patient to Available status.
    """
    sql = "UPDATE beds SET status = 'Available', occupied_by = NULL, admission_date = NULL WHERE occupied_by = ?"
    conn = get_connection(db_path)
    try:
        conn.execute(sql, (patient_id,))
        conn.commit()
    finally:
        conn.close()

def get_bed_status_stats(db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Returns metrics on general bed capacity and occupancy.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM beds")
        total = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM beds WHERE status = 'Occupied'")
        occupied = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM beds WHERE status = 'Available'")
        available = cursor.fetchone()[0]
        
        pct = (occupied / max(1, total)) * 100
        return {
            "total_beds": total,
            "occupied_beds": occupied,
            "available_beds": available,
            "occupancy_percentage": round(pct, 1)
        }
    finally:
        conn.close()

def get_ward_bed_stats(db_path: str = DB_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Returns metrics and available bed lists for all 10 predefined wards.
    """
    wards = [
        "Cardiology", "Neurology", "Orthopedics", "Emergency", "ICU",
        "General Medicine", "Pulmonology", "Nephrology", "Pediatrics", "Surgery"
    ]
    stats = {}
    conn = get_connection(db_path)
    try:
        for w in wards:
            cursor = conn.execute("SELECT COUNT(*) FROM beds WHERE ward = ?", (w,))
            total = cursor.fetchone()[0]
            cursor = conn.execute("SELECT COUNT(*) FROM beds WHERE ward = ? AND status = 'Occupied'", (w,))
            occupied = cursor.fetchone()[0]
            cursor = conn.execute("SELECT COUNT(*) FROM beds WHERE ward = ? AND status = 'Available'", (w,))
            available = cursor.fetchone()[0]
            
            # Retrieve list of available beds in this ward
            cursor_avail = conn.execute("SELECT bed_number FROM beds WHERE ward = ? AND status = 'Available' ORDER BY bed_number ASC", (w,))
            avail_beds = [r["bed_number"] for r in cursor_avail.fetchall()]
            
            stats[w] = {
                "total_capacity": total if total > 0 else 100,
                "occupied_count": occupied,
                "available_count": available,
                "available_beds": avail_beds
            }
        return stats
    finally:
        conn.close()

def get_department_distribution(db_path: str = DB_PATH) -> Dict[str, int]:
    """
    Returns the distribution of admitted patients by department.
    """
    sql = "SELECT department, COUNT(*) as count FROM patients WHERE current_status != 'Discharged' AND department IS NOT NULL AND department != '' GROUP BY department"
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return {row["department"]: row["count"] for row in rows}
    finally:
        conn.close()

def get_recent_admissions(limit: int = 5, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Returns the most recently registered active patient admissions.
    """
    sql = "SELECT * FROM patients WHERE current_status != 'Discharged' ORDER BY date_of_admission DESC, patient_id DESC LIMIT ?"
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(sql, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_next_patient_id(db_path: str = DB_PATH) -> str:
    """
    Generates the next sequential Patient ID.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT patient_id FROM patients ORDER BY LENGTH(patient_id) DESC, patient_id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            max_id = row["patient_id"]
            if max_id.startswith("P") and max_id[1:].isdigit():
                num = int(max_id[1:])
                return f"P{num+1:03d}"
        return "P001"
    finally:
        conn.close()
