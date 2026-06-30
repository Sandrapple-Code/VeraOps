import streamlit as st
import os
import numpy as np

# Import backend modules
from db.sqlite import add_patient, get_patient, get_all_patients
from llm.groq_client import generate_response

# Import RAG/Vector Store modules
from rag.document_store import create_doc_embeddings, save_faiss_index
from rag.patient_store import create_patient_text, embed_patient, add_patient_vector
from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.rag_pipeline import answer_query

st.set_page_config(page_title="VeraOps", layout="wide")

st.title("VeraOps - AI Hospital Assistant")

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
menu = st.sidebar.selectbox(
    "Choose Module",
    ["Dashboard", "Add Patient", "View Patient", "AI Assistant", "Upload SOPs"]
)

# -----------------------------
# DASHBOARD
# -----------------------------
if menu == "Dashboard":
    st.subheader("System Overview")

    patients = get_all_patients()

    # Layout with metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Registered Patients", len(patients))
    with col2:
        st.info("VeraOps backend is connected (SQLite + LLM + FAISS ready)")
        
    if patients:
        st.subheader("Recent Patient Records")
        # Convert list of rows/dicts to a readable format
        patient_list = [dict(p) for p in patients]
        st.dataframe(patient_list)

# -----------------------------
# ADD PATIENT
# -----------------------------
elif menu == "Add Patient":
    st.subheader("Add New Patient")

    with st.form("patient_form"):
        patient_id = st.text_input("Patient ID (e.g., P001)")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        diagnosis = st.text_input("Diagnosis")
        medicines = st.text_input("Medicines")
        ward = st.text_input("Ward")
        bed_number = st.text_input("Bed Number")
        visit_notes = st.text_area("Visit Notes")

        submit = st.form_submit_button("Save Patient")

        if submit:
            if not patient_id.strip():
                st.error("Patient ID is required.")
            else:
                data = {
                    "patient_id": patient_id.strip(),
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "diagnosis": diagnosis,
                    "medicines": medicines,
                    "ward": ward,
                    "bed_number": bed_number,
                    "visit_notes": visit_notes
                }

                try:
                    # 1. Add to SQLite Database
                    add_patient(data)
                    
                    # 2. Add to Patient FAISS Vector Store for clinical similarity matching
                    patient_text = create_patient_text(data)
                    patient_vector = embed_patient(patient_text)
                    add_patient_vector(data["patient_id"], patient_vector, patient_text)
                    
                    st.success(f"Patient '{name}' (ID: {patient_id}) added successfully to Database and Vector Store!")
                except Exception as e:
                    st.error(f"Error saving patient: {e}")

# -----------------------------
# VIEW PATIENT
# -----------------------------
elif menu == "View Patient":
    st.subheader("Search Patient")

    pid = st.text_input("Enter Patient ID")

    if st.button("Fetch"):
        if pid.strip():
            patient = get_patient(pid.strip())
            if patient:
                st.json(dict(patient))
            else:
                st.error("Patient not found")
        else:
            st.warning("Please enter a patient ID.")

# -----------------------------
# AI ASSISTANT
# -----------------------------
elif menu == "AI Assistant":
    st.subheader("VeraOps AI Assistant")

    api_key = st.text_input("Enter Groq API Key", type="password")
    
    # Optional Patient ID input
    patient_id = st.text_input("Enter Patient ID (Optional)")

    query = st.text_area("Ask a clinical or operational question:")

    if st.button("Ask VeraOps"):
        if not api_key:
            st.warning("Please enter Groq API key")
        elif not query.strip():
            st.warning("Please enter a query")
        else:
            with st.spinner("VeraOps is analyzing guidelines, matching cases, and generating an answer..."):
                try:
                    result = answer_query(api_key, query, patient_id)
                    
                    st.subheader("Generated Answer")
                    st.write(result["answer"])
                    
                    # Explanations / retrieved sources
                    st.write("---")
                    st.subheader("Reference & Explainability Sources")
                    
                    # Display structured patient if retrieved
                    if result.get("patient_record"):
                        with st.expander("Current Patient Structured Record"):
                            st.json(result["patient_record"])
                            
                    # Display retrieved SOP documents
                    with st.expander(f"Retrieved Hospital Guidelines (SOPs) - Count: {len(result['documents'])}"):
                        if result["documents"]:
                            for idx, doc in enumerate(result["documents"], 1):
                                st.markdown(f"**Source Chunk {idx} (L2 Distance: {doc['distance']:.4f})**")
                                st.write(doc["text"])
                                st.write("-" * 20)
                        else:
                            st.write("No matching hospital guidelines found.")
                            
                    # Display retrieved similar patients
                    with st.expander(f"Retrieved Similar Historical Cases - Count: {len(result['patients'])}"):
                        if result["patients"]:
                            for idx, pat in enumerate(result["patients"], 1):
                                st.markdown(f"**Case {idx} (Patient ID: {pat['patient_id']} | L2 Distance: {pat['distance']:.4f})**")
                                st.text(pat["text"])
                                st.write("-" * 20)
                        else:
                            st.write("No similar historical cases found.")
                            
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")

# -----------------------------
# UPLOAD SOPS
# -----------------------------
elif menu == "Upload SOPs":
    st.subheader("Upload Hospital SOPs / Protocols")
    
    uploaded_file = st.file_uploader("Upload SOP PDF", type=["pdf"])
    
    if uploaded_file is not None:
        temp_path = "temp_sop.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.info("Extracting text and generating embeddings...")
        try:
            # Load and chunk
            raw_text = load_pdf(temp_path)
            chunks = chunk_text(raw_text)
            
            # Index chunks
            create_doc_embeddings(chunks)
            save_faiss_index()
            
            st.success(f"Successfully indexed {len(chunks)} text segments from the SOP document!")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)