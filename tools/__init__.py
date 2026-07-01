from tools.patient_lookup import patient_lookup
from tools.hospital_knowledge import hospital_knowledge_search
from tools.patient_history import patient_history_search
from tools.similar_case import similar_case_search
from tools.bed_availability import bed_availability_lookup
from tools.register_patient import register_patient
from tools.update_patient import modify_patient_record

# Export all tools in a clean list
all_hospital_tools = [
    patient_lookup,
    hospital_knowledge_search,
    patient_history_search,
    similar_case_search,
    bed_availability_lookup,
    register_patient,
    modify_patient_record
]
