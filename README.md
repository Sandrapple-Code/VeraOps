# 🏥 VeraOps - Agentic AI Hospital Assistant

VeraOps is an AI-driven hospital intelligence system designed to streamline clinical workflows by integrating patient record management, retrieval-augmented generation (RAG), and LLM-based reasoning. It enables semantic access to medical protocols and patient history using vector search (FAISS) and supports modular AI orchestration for real-world healthcare decision support. Built with Python, Streamlit, and LLM APIs (Groq/Gemini).

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Agentic-green?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-purple?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue?style=for-the-badge&logo=sqlite)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM-black?style=for-the-badge)

### **An Intelligent Agentic RAG Powered Hospital Management & Clinical Decision Support System**

*Built using LangGraph, LangChain, FAISS, SQLite, Groq LLM and Streamlit.*

</div>

---

# 📌 Overview

VeraOps is an **Agentic AI Hospital Assistant** designed to bridge traditional Hospital Information Systems (HIS) with modern **Retrieval-Augmented Generation (RAG)** and **Agentic AI**.

Unlike conventional chatbots, VeraOps integrates:

- 🏥 Hospital Management
- 👨‍⚕️ Doctor Workspace
- 🧠 Agentic AI
- 📚 Retrieval-Augmented Generation (RAG)
- 📄 Automatic Clinical Documentation
- 🔎 Similar Patient Retrieval
- 🗄️ Vector Database Search
- 💬 Context-aware AI Conversations

to assist healthcare professionals with grounded, explainable and context-aware responses.

---

# ✨ Key Features

## 🏥 Hospital Information System

- Patient Registration
- Patient Directory
- Doctor Workspace
- Department Management
- Ward Management
- Automatic Bed Allocation
- Automatic Bed Release
- Hospital Dashboard
- Patient Search
- Dynamic Hospital Analytics

---

## 🤖 Agentic AI

Powered by **LangGraph**

The AI Agent intelligently decides which tools to use based on user queries.

Capabilities include:

- Patient Information Retrieval
- Hospital Knowledge Retrieval
- Similar Patient Search
- Clinical Question Answering
- AI Clinical Summary Generation
- Explainable Responses
- Conversation Memory

---

## 📚 Retrieval-Augmented Generation (RAG)

Two independent knowledge bases power VeraOps.

### Hospital Knowledge Base

Contains:

- SOPs
- Clinical Guidelines
- Drug Policies
- Laboratory References
- Infection Control
- Hospital Policies
- Medical Equipment Manuals

---

### Patient Knowledge Base

Every patient automatically gets a dedicated folder containing:

- Patient Summary
- Admission Report
- Doctor Notes
- Prescription
- Laboratory Reports
- Radiology Reports
- Treatment Plan
- Discharge Summary
- AI Clinical Summary

Whenever a patient record changes:

- Markdown documents are updated.
- Embeddings are regenerated.
- FAISS vectors are refreshed.
- AI immediately retrieves the latest information.

### Note - Whatever knowledge base has been used in this project is synthetic and does not include any real information.

---

# 🧠 Agentic Workflow

Instead of simply querying an LLM, VeraOps follows an intelligent workflow:

```
User Query
      │
      ▼
LangGraph Agent
      │
      ├───────────────┐
      │               │
      ▼               ▼
Patient Tools     Hospital Tools
      │               │
      └──────┬────────┘
             ▼
      Similar Patient Search
             ▼
      Context Aggregation
             ▼
        Groq LLM
             ▼
 Grounded AI Response
             ▼
 Source Citations
```

---

# 🏗️ System Architecture

```
                     Streamlit UI
                          │
 ┌────────────────────────┼────────────────────────┐
 │                        │                        │
 ▼                        ▼                        ▼
Dashboard         Doctor Workspace         AI Assistant
 │                        │                        │
 └──────────────┬─────────┴───────────────┬────────┘
                ▼                         ▼
         SQLite Database          Hospital Knowledge
                │                         │
                ▼                         ▼
      Patient Documents           Hospital Documents
                │                         │
                └────────────┬────────────┘
                             ▼
                   Embedding Model
                             ▼
                     FAISS Vector Store
                             ▼
                     LangGraph Agent
                             ▼
                         Groq LLM
                             ▼
                     AI Generated Response
```

---

# 📂 Project Structure

```
VeraOps/

│── app.py
│── requirements.txt
│── README.md

├── agents/
│
├── db/
│   ├── sqlite.py
│   └── hospital.db
│
├── data/
│   └── patients.json
│
├── patient_documents/
│
├── hospital_docs/
│
├── vector_store/
│
├── scripts/
│
├── tools/
│
├── assets/
│
├── pages/
│
└── utils/
```

---

# 👨‍⚕️ Doctor Workspace

The Doctor Workspace enables clinicians to update patient information through a structured workflow.

### Patient Summary

Displays:

- Personal Information
- Admission Details
- Ward
- Bed
- Diagnosis
- Assigned Doctor

### Doctor Notes

Doctors can continuously append observations.

Automatically updates:

```
doctor_notes.md
```

---

### Prescription

Maintain medication history.

Automatically updates:

```
prescription.md
```

---

### Laboratory

Store laboratory investigations.

Updates:

```
lab_report.md
```

---

### Radiology

Maintain:

- CT
- MRI
- X-Ray
- Ultrasound

Updates:

```
radiology_report.md
```

---

### Treatment Plan

Updates:

```
treatment_plan.md
```

---

### Patient Timeline

Maintains chronological events:

- Registration
- Admission
- Doctor Notes
- Prescription
- Lab Updates
- Radiology Updates
- AI Summary
- Discharge

---

# 🛏️ Bed Management

Every ward contains **100 beds**.

The system automatically:

- Allocates available beds
- Prevents duplicate assignment
- Releases beds on discharge
- Calculates occupancy
- Updates dashboard analytics

---

# 🔎 Similar Patient Search

Uses semantic embeddings to retrieve patients with similar:

- Diagnosis
- Symptoms
- Treatment
- Medical History

Helping doctors compare previous cases.

---

# 💬 AI Assistant

Supports natural language questions such as:

- Summarize patient P012.
- Show latest doctor notes.
- Find similar heart failure cases.
- Explain hospital hypertension guideline.
- What treatment is the patient currently receiving?
- What medications were prescribed?

---

# 📊 Dashboard

The dashboard displays:

- Total Patients
- Admissions
- Discharges
- Hospital Capacity
- Occupied Beds
- Available Beds
- Department Distribution
- Ward Occupancy
- Recent Activities

---

# ℹ️ About VeraOps

Displays live project information:

- Embedding Model
- Vector Database
- Total Hospital Documents
- Total Patient Documents
- Vector Count
- RAG Pipeline
- Database Statistics
- Knowledge Base Statistics

---

# 🧩 Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Frontend | Streamlit |
| Database | SQLite |
| Vector Store | FAISS |
| Agent Framework | LangGraph |
| LLM Framework | LangChain |
| Embeddings | Sentence Transformers |
| LLM | Groq |
| Document Parsing | PyMuPDF |
| Data Format | Markdown, PDF, DOCX, TXT |

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/VeraOps.git
```

Move into project

## Setup
1. Copy `settings.example.json` to `settings.json`
2. Get a free API key from https://console.groq.com/keys
3. Paste your key into `settings.json` as `groq_api_key`
4. Run the app

```bash
cd VeraOps
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

Install packages

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

# 🔑 API Key

VeraOps allows users to securely provide their own **Groq API Key** directly from the application.

No API key is stored permanently.

---

# 📖 RAG Pipeline

```
Hospital Documents
          │
Patient Documents
          │
          ▼
Document Loader
          ▼
Text Chunking
          ▼
Embeddings
          ▼
FAISS Index
          ▼
Retriever
          ▼
LangGraph Agent
          ▼
Groq LLM
          ▼
Grounded Response
```

---

# 🎯 Future Improvements

- Hybrid Search (BM25 + Semantic)
- Authentication
- Multi-user Support
- Cloud Database
- HL7/FHIR Integration
- PACS Integration
- Real-time Notifications
- Voice-enabled Clinical Assistant

---

# 👨‍💻 Developer

**Sanskriti Shakya**

Developed as an Agentic AI Healthcare project integrating:

- LangGraph
- LangChain
- FAISS
- SQLite
- Groq
- Streamlit

---

# ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational and research purposes only.

It is **not certified for real-world clinical deployment** and should not be used to make medical decisions without qualified healthcare professionals.
