import json
from langchain_core.tools import tool
from db.sqlite import get_ward_bed_stats

@tool
def bed_availability_lookup() -> str:
    """
    Retrieves the current bed occupancy and lists remaining available beds by ward.
    
    :return: JSON formatted string containing capacities, occupied, and available beds.
    """
    try:
        stats = get_ward_bed_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        return f"Error retrieving bed availability: {str(e)}"
