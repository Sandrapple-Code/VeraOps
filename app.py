import streamlit as st
import os
import numpy as np

# Import backend modules
from db.sqlite import add_patient, get_patient, get_all_patients
from llm.groq_client import generate_response

# Import RAG/Vector Store modules
from rag.document_store import search_docs, create_doc_embeddings, save_faiss_index
from rag.patient_store import (
    create_patient_text,
    embed_patient,
    add_patient_vector,
    search_similar_patients
)
from rag.loader import load_pdf
from rag.chunker import chunk_text

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

    # Options to use RAG references
    use_rag_docs = st.checkbox("Include Hospital Guidelines (SOPs)", value=True)
    use_rag_patients = st.checkbox("Find Similar Patient Cases", value=True)

    query = st.text_area("Ask a clinical or operational question:")

    if st.button("Generate Response"):
        if not api_key:
            st.warning("Please enter Groq API key")
        elif not query.strip():
            st.warning("Please enter a query")
        else:
            context_docs = ""
            context_patients = ""
            
            # Retrieve relevant SOP documents
            if use_rag_docs:
                try:
                    doc_results = search_docs(query, k=2)
                    if doc_results:
                        context_docs = "\n\nRelevant Hospital Guidelines / SOPs:\n" + "\n".join(
                            [f"- {r['text']}" for r in doc_results]
                        )
                except Exception as e:
                    st.warning(f"Could not retrieve SOP guidelines: {e}")
            
            # Retrieve similar cases
            if use_rag_patients:
                try:
                    patient_results = search_similar_patients(query, k=2)
                    if patient_results:
                        context_patients = "\n\nSimilar Clinical Cases:\n" + "\n".join(
                            [f"- {r['text']}" for r in patient_results]
                        )
                except Exception as e:
                    st.warning(f"Could not retrieve similar clinical cases: {e}")
            
            # Construct final prompt with retrieved context
            final_prompt = query
            if context_docs or context_patients:
                final_prompt = (
                    f"You are an AI Hospital Assistant. Use the following context to help answer the user query.\n"
                    f"{context_docs}"
                    f"{context_patients}\n\n"
                    f"User Query: {query}\n"
                    f"Assistant Answer:"
                )
            
            with st.spinner("Generating answer from Groq..."):
                try:
                    response = generate_response(api_key, final_prompt)
                    st.subheader("Response")
                    st.write(response)
                    
                    # Collapsible context details
                    if context_docs or context_patients:
                        with st.expander("Show Retrieved Reference Context"):
                            if context_docs:
                                st.write("**Guidelines Context:**", context_docs)
                            if context_patients:
                                st.write("**Similar Cases Context:**", context_patients)
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