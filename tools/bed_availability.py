import json
from langchain_core.tools import tool
from db.sqlite import get_connection

@tool
def bed_availability_lookup() -> str:
    """
    Retrieves the current bed occupancy and lists remaining available beds by ward.
    
    :return: JSON formatted string containing capacities, occupied, and available beds.
    """
    # Predefined bed capacities across hospital departments
    capacities = {
        "Cardiovascular Sub-Acute Unit": ["C-301", "C-302", "C-303", "C-304", "C-305"],
        "Surgical Ward B": ["B-101", "B-102", "B-103", "B-104", "B-105"],
        "Stroke Unit": ["S-201", "S-202", "S-203", "S-204", "S-205"],
        "ICU": ["ICU-01", "ICU-02", "ICU-03"],
        "General Ward A": ["A-101", "A-102", "A-103", "A-104", "A-105", "A-106", "A-107", "A-108", "A-109", "A-110"]
    }
    
    try:
        conn = get_connection()
        try:
            # Query all patients occupying a bed (ward and bed_number are assigned)
            cursor = conn.execute(
                "SELECT ward, bed_number FROM patients WHERE ward IS NOT NULL AND ward != '' AND bed_number IS NOT NULL AND bed_number != ''"
            )
            occupied_rows = cursor.fetchall()
        finally:
            conn.close()
            
        occupied_beds = {}
        for row in occupied_rows:
            w = row["ward"].strip()
            b = row["bed_number"].strip()
            if w not in occupied_beds:
                occupied_beds[w] = set()
            occupied_beds[w].add(b)
            
        availability = {}
        for ward, beds in capacities.items():
            occupied_in_ward = occupied_beds.get(ward, set())
            available_beds = [b for b in beds if b not in occupied_in_ward]
            availability[ward] = {
                "total_capacity": len(beds),
                "occupied_count": len(occupied_in_ward),
                "available_count": len(available_beds),
                "available_beds": available_beds
            }
            
        # Report any other wards not listed in capacities
        for w, beds in occupied_beds.items():
            if w not in capacities:
                availability[w] = {
                    "total_capacity": "Unknown",
                    "occupied_count": len(beds),
                    "available_count": "Unknown",
                    "occupied_beds": list(beds)
                }
                
        return json.dumps(availability, indent=2)
    except Exception as e:
        return f"Error retrieving bed availability: {str(e)}"
