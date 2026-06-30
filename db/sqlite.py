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
    Initializes the database schema by creating the patients table if it doesn't exist.
    
    :param db_path: Path to the SQLite database file.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        name TEXT,
        age INTEGER,
        gender TEXT,
        diagnosis TEXT,
        medicines TEXT,
        ward TEXT,
        bed_number TEXT,
        visit_notes TEXT
    );
    """
    conn = get_connection(db_path)
    try:
        conn.execute(create_table_sql)
        conn.commit()
    finally:
        conn.close()

# Automatically initialize database when the module is imported
init_db()

def add_patient(data: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """
    Adds a new patient to the database.
    
    :param data: Dictionary containing patient details. Must contain 'patient_id'.
    :param db_path: Path to the SQLite database file.
    """
    if "patient_id" not in data:
        raise ValueError("Missing 'patient_id' in patient data")
        
    fields = [
        "patient_id", "name", "age", "gender", "diagnosis", 
        "medicines", "ward", "bed_number", "visit_notes"
    ]
    
    # Extract values matching the expected fields, defaulting to None if missing
    values = [data.get(field) for field in fields]
    
    placeholders = ", ".join(["?"] * len(fields))
    columns = ", ".join(fields)
    sql = f"INSERT INTO patients ({columns}) VALUES ({placeholders})"
    
    conn = get_connection(db_path)
    try:
        conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()

def get_patient(patient_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single patient by their patient_id.
    
    :param patient_id: ID of the patient to retrieve.
    :param db_path: Path to the SQLite database file.
    :return: Dictionary of patient details, or None if not found.
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
    
    :param patient_id: ID of the patient to update.
    :param data: Dictionary containing fields to update.
    :param db_path: Path to the SQLite database file.
    :return: True if the patient was updated successfully, False otherwise.
    """
    # Exclude patient_id from the update fields to avoid modifying the primary key
    update_data = {k: v for k, v in data.items() if k != "patient_id"}
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
    
    :param db_path: Path to the SQLite database file.
    :return: List of dictionaries representing all patients.
    """
    sql = "SELECT * FROM patients"
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

if __name__ == "__main__":
    # Test suite to verify standalone execution
    print("Initializing test database...")
    test_db = "hospital_test.db"
    
    # Clean up test database if it exists from a previous run
    if os.path.exists(test_db):
        os.remove(test_db)
        
    init_db(test_db)
    print(f"Database table created in {test_db}.")
    
    # 1. Test add_patient
    patient_data = {
        "patient_id": "P001",
        "name": "Alice Smith",
        "age": 34,
        "gender": "Female",
        "diagnosis": "Acute Appendicitis",
        "medicines": "Ibuprofen, Amoxicillin",
        "ward": "General Ward A",
        "bed_number": "A-12",
        "visit_notes": "Patient reports abdominal pain. Prepared for surgery."
    }
    
    print("\nTesting add_patient...")
    add_patient(patient_data, db_path=test_db)
    print("Added patient P001 successfully.")
    
    # 2. Test get_patient
    print("\nTesting get_patient...")
    p = get_patient("P001", db_path=test_db)
    print("Retrieved patient:", p)
    assert p is not None
    assert p["name"] == "Alice Smith"
    
    # 3. Test update_patient
    print("\nTesting update_patient...")
    update_data = {
        "age": 35,
        "ward": "ICU",
        "bed_number": "ICU-03",
        "visit_notes": "Post-surgery recovery stable."
    }
    updated = update_patient("P001", update_data, db_path=test_db)
    print("Update successful?", updated)
    p_updated = get_patient("P001", db_path=test_db)
    print("Retrieved updated patient:", p_updated)
    assert p_updated["age"] == 35
    assert p_updated["ward"] == "ICU"
    
    # 4. Test get_all_patients
    print("\nTesting get_all_patients...")
    # Add another patient
    patient_data_2 = {
        "patient_id": "P002",
        "name": "Bob Jones",
        "age": 45,
        "gender": "Male",
        "diagnosis": "Hypertension"
    }
    add_patient(patient_data_2, db_path=test_db)
    all_p = get_all_patients(db_path=test_db)
    print("Total patients in database:", len(all_p))
    for patient in all_p:
        print(f"- {patient['patient_id']}: {patient['name']} ({patient['diagnosis'] or 'No diagnosis'})")
        
    assert len(all_p) == 2
    
    # Clean up test database
    if os.path.exists(test_db):
        os.remove(test_db)
        
    print("\nAll sqlite database CRUD tests passed successfully!")
