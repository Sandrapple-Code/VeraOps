import streamlit as st
import os
import numpy as np

# Import backend modules
from db.sqlite import add_patient, get_patient, get_all_patients
from llm.groq_client import generate_response

# Import RAG/Vector Store modules
import json
import importlib
import rag.loader
importlib.reload(rag.loader)

from rag.patient_store import create_patient_text, embed_patient, add_patient_vector
from rag.ingestion import check_needs_rebuild, rebuild_knowledge_base, get_knowledge_base_stats
from agents import run_agent

st.set_page_config(page_title="VeraOps", layout="wide")

st.title("VeraOps - AI Hospital Assistant")

# Project root path helper
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Automatically check and build/refresh indices on startup if needed
if "index_checked" not in st.session_state:
    if check_needs_rebuild():
        with st.spinner("Initializing/updating vector databases on startup..."):
            try:
                rebuild_knowledge_base()
            except Exception as e:
                st.error(f"Failed to auto-rebuild vector stores on startup: {e}")
    st.session_state.index_checked = True

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
menu = st.sidebar.selectbox(
    "Choose Module",
    ["Dashboard", "Add Patient", "View Patient", "AI Assistant", "Knowledge Base"]
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
            # Initialize session state chat history if missing
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
                
            with st.spinner("VeraOps agent is reasoning, selecting tools, and processing guidelines..."):
                try:
                    # Run the LangGraph agent
                    result = run_agent(query, st.session_state.chat_history, api_key)
                    
                    # Update local session history
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    st.session_state.chat_history.append({"role": "assistant", "content": result["final_response"]})
                    
                    st.subheader("Generated Answer")
                    st.markdown(result["final_response"])
                    
                    # Explainability & Observability details
                    st.write("---")
                    st.subheader("Reference & Explainability Sources")
                    
                    # 1. Tool execution sequence summary
                    with st.expander(f"Tool Execution Logs - Count: {len(result['tool_runs'])}"):
                        if result["tool_runs"]:
                            for idx, run in enumerate(result["tool_runs"], 1):
                                st.markdown(f"**Step {idx}: Executed `{run['tool']}`**")
                                st.code(f"Arguments: {json.dumps(run['args'], indent=2)}")
                                st.markdown("**Output Preview:**")
                                st.write(run["result_summary"])
                                st.write("-" * 20)
                        else:
                            st.write("No tools were required to answer this query.")
                            
                    # 2. List of cited resources
                    with st.expander(f"Cited Resources - Count: {len(result['sources'])}"):
                        if result["sources"]:
                            for s in result["sources"]:
                                st.markdown(f"- **{s}**")
                        else:
                            st.write("No database records or clinical documents were explicitly cited.")
                            
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")

# -----------------------------
# KNOWLEDGE BASE
# -----------------------------
elif menu == "Knowledge Base":
    st.subheader("VeraOps Knowledge Base & Indexing")
    
    # Show current knowledge base statistics
    stats = get_knowledge_base_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hospital SOP Files", stats["hospital_docs_count"])
        st.metric("Hospital Chunk Count", stats["hospital_vector_count"])
    with col2:
        st.metric("Patient MD Files", stats["patient_docs_count"])
        st.metric("Patient Chunk Count", stats["patient_vector_count"])
    with col3:
        st.metric("Total Indexed Chunks", stats["total_chunks"])
        st.info(f"Embedding Model: **{stats['embedding_model']}**")
        
    st.write("---")
    
    # SOP File Uploader
    st.markdown("### Upload new Hospital SOPs / Protocols")
    uploaded_file = st.file_uploader("Choose a PDF or Markdown file to add to hospital_docs/", type=["pdf", "md"])
    if uploaded_file is not None:
        save_dir = os.path.join(PROJECT_ROOT, "hospital_docs")
        os.makedirs(save_dir, exist_ok=True)
        dest_path = os.path.join(save_dir, uploaded_file.name)
        
        try:
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Successfully uploaded '{uploaded_file.name}' to hospital_docs/.")
            # Trigger warning check
            if check_needs_rebuild():
                st.info("New document detected. Rebuilding the Knowledge Base is recommended to index it.")
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
            
    st.write("---")
    
    st.markdown("### Rebuild Vector Store")
    st.write("Rebuilding will clear the existing FAISS indexes, scan all PDF/MD files in `hospital_docs/` and patient records in `patient_documents/`, and generate fresh embeddings.")
    
    # Warning check
    if check_needs_rebuild():
        st.warning("⚠️ Changes detected (or index files are missing). Rebuilding the Knowledge Base is recommended!")
        
    if st.button("Rebuild Knowledge Base"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        def update_progress(percent: float, text: str):
            progress_bar.progress(percent)
            status_text.write(text)
            
        with st.spinner("Rebuilding knowledge base indices..."):
            try:
                h_chunks, p_chunks = rebuild_knowledge_base(update_progress)
                st.success(f"Successfully rebuilt! Indexed {h_chunks} hospital chunks and {p_chunks} patient chunks.")
                # Force refresh of page stats
                st.rerun()
            except Exception as e:
                st.error(f"Rebuild failed: {e}")