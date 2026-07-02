import streamlit as st
import os
import sys
import json
import glob
from datetime import datetime
from typing import Dict, Any, List

# Ensure the project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import backend modules
from db.sqlite import (
    get_all_patients,
    get_patient,
    get_available_beds,
    get_bed_status_stats,
    get_ward_bed_stats,
    get_department_distribution,
    get_recent_admissions,
    get_next_patient_id,
    release_bed,
    occupy_bed,
    update_patient,
    add_timeline_event,
    get_patient_timeline
)
from llm.groq_client import generate_response, get_groq_client
from rag.ingestion import check_needs_rebuild, rebuild_knowledge_base, get_knowledge_base_stats, index_patient_documents
from rag.patient_store import search_similar_patients
from agents import run_agent
from services.patient_service import register_new_patient
from services.config_service import load_settings, save_settings, get_setting, set_setting
from tools.bed_availability import bed_availability_lookup
from tools.update_patient import modify_patient_record

# Streamlit Page Config
st.set_page_config(page_title="VeraOps - AI Hospital Assistant", layout="wide", page_icon="🏥")

# Load settings
settings = load_settings()
api_key = settings.get("groq_api_key", "")
model_name = settings.get("model_selection", "llama-3.3-70b-versatile")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SESSION_FILE = os.path.join(PROJECT_ROOT, "chat_history_sessions.json")

# Helper functions for persistent chat sessions
def load_chat_sessions() -> Dict[str, Any]:
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_chat_sessions(sessions: Dict[str, Any]):
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to persist chat history: {e}")

# Helper for recent AI activities
def get_recent_ai_activity(limit: int = 5) -> List[Dict[str, Any]]:
    sessions = load_chat_sessions()
    activities = []
    for sess_id, sess in sessions.items():
        messages = sess.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                activities.append({
                    "query": msg.get("content", ""),
                    "session_title": sess.get("title", "New Chat")
                })
    return activities[-limit:][::-1]

# -----------------------------
# AUTO-SEEDING & SYSTEM INITIALIZATION
# -----------------------------
if "index_checked" not in st.session_state:
    # Check if database is empty
    try:
        patients_check = get_all_patients()
    except Exception:
        patients_check = []
        
    if not patients_check:
        with st.spinner("Initializing and seeding database with default patient records..."):
            try:
                from scripts.seed_database import seed_database
                json_file = os.path.join(PROJECT_ROOT, "data", "patients.json")
                seed_database(json_file)
            except Exception as e:
                st.error(f"Failed to auto-seed database: {e}")
                
    # Check if vector indices need rebuilding
    if check_needs_rebuild():
        with st.spinner("Initializing/updating vector databases on startup..."):
            try:
                rebuild_knowledge_base()
            except Exception as e:
                st.error(f"Failed to auto-rebuild vector stores on startup: {e}")
    st.session_state.index_checked = True

# Initialize Session State login & navigation variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"

# Initialize persistent chat sessions
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()
if "active_session_id" not in st.session_state:
    if st.session_state.chat_sessions:
        st.session_state.active_session_id = list(st.session_state.chat_sessions.keys())[0]
    else:
        new_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        st.session_state.chat_sessions[new_id] = {
            "title": "New Chat Thread",
            "messages": []
        }
        st.session_state.active_session_id = new_id

# -----------------------------
# AUTHENTICATION GATE & LANDING PAGE
# -----------------------------
if not st.session_state.authenticated:
    st.markdown("""
        <style>
        /* Modern clinical theme background with smooth gradient */
        .stApp {
            background: linear-gradient(135deg, #0F172A 0%, #111827 50%, #0F2A2A 100%) !important;
            color: #F8FAFC !important;
            font-family: 'Outfit', 'Inter', sans-serif;
        }
        /* Fancy Gradient headings */
        h1, h2, h3, h4, h5, h6 {
            background: linear-gradient(to right, #38BDF8, #34D399, #10B981) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            font-family: 'Outfit', 'Inter', sans-serif !important;
            font-weight: 900 !important;
            margin-bottom: 12px !important;
        }
        .hero-title {
            font-size: 4.5rem;
            font-weight: 900;
            background: linear-gradient(to right, #38BDF8, #34D399, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-top: 0.5rem;
            margin-bottom: 0.2rem;
            filter: drop-shadow(0 2px 8px rgba(56, 189, 248, 0.3));
            transition: all 0.5s ease;
        }
        .hero-title:hover {
            transform: scale(1.02);
            filter: drop-shadow(0 4px 20px rgba(16, 185, 129, 0.6));
        }
        .hero-motto {
            font-size: 1.6rem;
            color: #94A3B8;
            text-align: center;
            margin-bottom: 3rem;
            font-style: italic;
            font-weight: 300;
        }
        .landing-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
        }
        .landing-card:hover {
            transform: translateY(-5px);
            border-color: #10B981;
            box-shadow: 0 12px 20px -3px rgba(16, 185, 129, 0.2), 0 4px 12px -2px rgba(16, 185, 129, 0.1);
        }
        .card-title {
            color: #38BDF8;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .card-desc {
            color: #CBD5E1;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        /* Style native containers to act as elegant, hoverable cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease, border-color 0.3s ease !important;
            border-radius: 10px !important;
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-3px);
            border-color: #10B981 !important;
            box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.2), 0 4px 6px -2px rgba(16, 185, 129, 0.1);
        }
        /* Custom Button overrides */
        div.stButton > button {
            transition: all 0.2s ease-in-out !important;
        }
        div.stButton > button:hover {
            transform: scale(1.02);
        }
        /* Landing page extra scrolling spacer */
        .scrolling-spacer {
            height: 120px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 1. Logo Centered above the heading
    lcol_1, lcol_2, lcol_3 = st.columns([1.5, 1, 1.5])
    with lcol_2:
        logo_path = os.path.join(PROJECT_ROOT, "data", "logo_transparent.png")
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
            
    st.markdown('<h1 class="hero-title">VeraOps</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-motto">Orchestrating Hospital Intelligence with Agentic Precision</p>', unsafe_allow_html=True)
    
    # Feature columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="landing-card">
                <div class="card-title">🤖 AI Clinical Agent</div>
                <div class="card-desc">LangGraph supervisor orchestrates specialized clinical and operational tools dynamically to guide healthcare delivery.</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="landing-card">
                <div class="card-title">📚 Double-RAG Search</div>
                <div class="card-desc">Parallel FAISS search vectors retrieve matching SOP protocols and historical patient case details instantly.</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="landing-card">
                <div class="card-title">🛏 Ward Planner</div>
                <div class="card-desc">Real-time bed allocation trackers and occupancy logs synchronized directly with SQL.</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="landing-card">
                <div class="card-title">📄 Clinical Timelines</div>
                <div class="card-desc">Consolidated profiles rendering admission, prescriptions, and lab notes in a structured patient course timeline.</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    st.write("")
    
    # Main container
    lcol, rcol = st.columns([1.1, 0.9])
    with lcol:
        st.write("### About VeraOps")
        st.write("""
        **VeraOps** is designed for hospital operational efficiency. By connecting direct database CRUD structures with advanced reasoning LLM pipelines, it solves key healthcare coordination issues:
        - **Semantic Knowledge Retrievals**: Access hospital policies, protocols, and directives without reading hundreds of pages of PDF SOPs.
        - **Patient Profiling**: Consolidate database fields and timeline documentation to give doctors an instantaneous view of a patient's historical course.
        - **Similar Case Intelligence**: Find similar diagnoses and symptoms within the vector database to support evidence-based medicine.
        """)
        st.info("🔑 **Portal Verification Credentials:**\n- **Hospital User ID**: `admin` \n- **Password**: `hospital`")
        
    with rcol:
        with st.container(border=True):
            st.markdown('<h3 style="text-align: center; color: #38BDF8; margin-top: 0; margin-bottom: 20px;">🏥 Secure Hospital Login Gate</h3>', unsafe_allow_html=True)
            
            user_id = st.text_input("Hospital User ID", key="login_uid")
            password = st.text_input("Password", type="password", key="login_pwd")
            
            if st.button("Authenticate & Open App", use_container_width=True):
                if user_id.strip() == "admin" and password == "hospital":
                    st.session_state.authenticated = True
                    st.success("Verification successful! Opening portal...")
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid User ID or Password.")
        
    # Spacer at the bottom to ensure comfortable scrolling
    st.markdown('<div class="scrolling-spacer"></div>', unsafe_allow_html=True)

# -----------------------------
# MAIN APP FLOW (AUTHENTICATED)
# -----------------------------
else:
    # Sidebar navigation buttons (clickable boxes)
    st.sidebar.markdown("### 🏥 VeraOps Health Center")
    
    # Theme configuration
    theme = st.sidebar.selectbox("🎨 App Theme", ["Dark Mode", "Light Mode"], index=0)
    
    # Global visual styling (soothing gradients, text-gradient headings and container transitions)
    if theme == "Dark Mode":
        st.markdown("""
            <style>
            /* Main background with classy slate-blue gradient */
            .stApp {
                background: linear-gradient(135deg, #0F172A 0%, #111827 50%, #0F2A2A 100%) !important;
                color: #E2E8F0;
                font-family: 'Inter', sans-serif;
            }
            /* Soothing medical text-gradient headings */
            h1, h2, h3, h4, h5, h6 {
                background: linear-gradient(to right, #38BDF8, #34D399, #10B981) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                font-family: 'Outfit', 'Inter', sans-serif !important;
                font-weight: 800 !important;
                margin-bottom: 12px !important;
            }
            /* Sidebar background with elegant gradient */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%) !important;
            }
            [data-testid="stSidebar"] * {
                color: #E2E8F0 !important;
            }
            /* Sidebar buttons box styling */
            [data-testid="stSidebar"] div.stButton > button {
                background-color: #1E293B !important;
                border: 1px solid #334155 !important;
                color: #CBD5E1 !important;
                text-align: left !important;
                padding: 10px 15px !important;
                font-size: 0.95rem !important;
                transition: all 0.2s ease-in-out !important;
                border-radius: 8px !important;
                display: block !important;
                width: 100% !important;
            }
            [data-testid="stSidebar"] div.stButton > button:hover {
                background-color: #10B981 !important;
                border-color: #10B981 !important;
                color: #0F172A !important;
                transform: translateX(4px);
            }
            /* Metric text color green override */
            div[data-testid="stMetricValue"] {
                color: #10B981 !important;
                font-weight: bold !important;
                transition: all 0.3s ease;
            }
            div[data-testid="stMetricLabel"] {
                color: #94A3B8 !important;
            }
            /* Style native containers to act as elegant, hoverable cards */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease, border-color 0.3s ease !important;
                border-radius: 10px !important;
                background-color: #1E293B !important;
                border: 1px solid #334155 !important;
            }
            div[data-testid="stVerticalBlockBorderWrapper"]:hover {
                transform: translateY(-3px);
                border-color: #10B981 !important;
                box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.2), 0 4px 6px -2px rgba(16, 185, 129, 0.1);
            }
            .card-header {
                color: #38BDF8;
                font-weight: bold;
                font-size: 1.1rem;
                margin-bottom: 8px;
            }
            /* Styling standard buttons */
            button[data-testid="baseButton-secondary"] {
                transition: all 0.2s ease-in-out !important;
            }
            button[data-testid="baseButton-secondary"]:hover {
                background-color: #10B981 !important;
                color: #0F172A !important;
                border-color: #10B981 !important;
                transform: scale(1.02);
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            /* Main background with classy mint-sky gradient */
            .stApp {
                background: linear-gradient(135deg, #F0FDF4 0%, #E0F2FE 50%, #F8FAFC 100%) !important;
                color: #0F172A;
                font-family: 'Inter', sans-serif;
            }
            /* Soothing medical text-gradient headings for Light Mode */
            h1, h2, h3, h4, h5, h6 {
                background: linear-gradient(to right, #0284C7, #0D9488, #16A34A) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                font-family: 'Outfit', 'Inter', sans-serif !important;
                font-weight: 800 !important;
                margin-bottom: 12px !important;
            }
            /* Sidebar background with elegant light gradient */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #F1F5F9 0%, #E2E8F0 100%) !important;
            }
            /* Sidebar buttons box styling */
            [data-testid="stSidebar"] div.stButton > button {
                background-color: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                color: #475569 !important;
                text-align: left !important;
                padding: 10px 15px !important;
                font-size: 0.95rem !important;
                transition: all 0.2s ease-in-out !important;
                border-radius: 8px !important;
                display: block !important;
                width: 100% !important;
            }
            [data-testid="stSidebar"] div.stButton > button:hover {
                background-color: #16A34A !important;
                border-color: #16A34A !important;
                color: #FFFFFF !important;
                transform: translateX(4px);
            }
            /* Metric text color green override */
            div[data-testid="stMetricValue"] {
                color: #16A34A !important;
                font-weight: bold !important;
                transition: all 0.3s ease;
            }
            div[data-testid="stMetricLabel"] {
                color: #475569 !important;
            }
            /* Style native containers to act as elegant, hoverable cards */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease, border-color 0.3s ease !important;
                border-radius: 10px !important;
                background-color: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            }
            div[data-testid="stVerticalBlockBorderWrapper"]:hover {
                transform: translateY(-3px);
                border-color: #16A34A !important;
                box-shadow: 0 10px 15px -3px rgba(22, 163, 74, 0.15), 0 4px 6px -2px rgba(22, 163, 74, 0.08);
            }
            .card-header {
                color: #0284C7;
                font-weight: bold;
                font-size: 1.1rem;
                margin-bottom: 8px;
            }
            /* Styling standard buttons */
            button[data-testid="baseButton-secondary"] {
                transition: all 0.2s ease-in-out !important;
            }
            button[data-testid="baseButton-secondary"]:hover {
                background-color: #16A34A !important;
                color: #FFFFFF !important;
                border-color: #16A34A !important;
                transform: scale(1.02);
            }
            </style>
        """, unsafe_allow_html=True)

    # Clickable boxes for sidebar navigation
    st.sidebar.markdown("### 🗺️ Navigation Menu")
    pages = [
        ("🏠 Dashboard", "🏠 Dashboard"),
        ("🤖 AI Assistant", "🤖 AI Assistant"),
        ("👤 Patient Registration", "👤 Patient Registration"),
        ("🩺 Doctor Workspace", "🩺 Doctor Workspace"),
        ("📋 Patient Directory", "📋 Patient Directory"),
        ("📚 Knowledge Base", "📚 Knowledge Base"),
        ("🛏 Ward & Bed Management", "🛏 Ward & Bed Management"),
        ("⚙ Settings", "⚙ Settings"),
        ("ℹ About VeraOps", "ℹ About VeraOps")
    ]
    
    for label, page_key in pages:
        is_active = (st.session_state.current_page == page_key)
        btn_label = f"🟢 {label}" if is_active else label
        if st.sidebar.button(btn_label, use_container_width=True, key=f"btn_{page_key}"):
            st.session_state.current_page = page_key
            st.rerun()
            
    menu = st.session_state.current_page

    # Log Out at the bottom
    st.sidebar.write("---")
    if st.sidebar.button("🔒 Log Out Portal", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    # -----------------------------
    # 1. DASHBOARD
    # -----------------------------
    if menu == "🏠 Dashboard":
        st.title("🏠 Hospital System Dashboard")
        st.markdown("Real-time clinical, operational, and database metrics.")
        
        # Calculate patient statistics
        patients = get_all_patients()
        total_patients = len(patients)
        
        # Count Today's Admissions & Discharges
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_admissions = len([p for p in patients if p.get("date_of_admission") == today_str])
        today_discharges = len([p for p in patients if p.get("date_of_discharge") == today_str])
        
        # Calculate bed statistics
        bed_stats = get_bed_status_stats()
        total_avail_beds = bed_stats["available_beds"]
        total_occ_beds = bed_stats["occupied_beds"]
        total_beds = bed_stats["total_beds"]
        occupancy_percentage = bed_stats["occupancy_percentage"]
        
        # Layout with metrics grid
        st.markdown("### Clinical Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Total Patients", total_patients)
        with m_col2:
            st.metric("Today's Admissions", today_admissions)
        with m_col3:
            st.metric("Today's Discharges", today_discharges)
            
        st.markdown("### Bed Management")
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.metric("Available Beds", total_avail_beds)
        with b_col2:
            st.metric("Occupied Beds", total_occ_beds)
        with b_col3:
            st.metric("Total Beds Capacity", total_beds)
        with b_col4:
            st.metric("Occupancy Rate", f"{occupancy_percentage}%")

        st.write("---")
        
        # Ward occupancy & Department distribution side by side
        col_w, col_d = st.columns(2)
        with col_w:
            with st.container(border=True):
                st.markdown('<div class="card-header">🚪 Ward Occupancy Breakdown</div>', unsafe_allow_html=True)
                ward_stats = get_ward_bed_stats()
                ward_rows = []
                for w, s in ward_stats.items():
                    tot = s["total_capacity"]
                    occ = s["occupied_count"]
                    av = s["available_count"]
                    pct = (occ / max(1, tot)) * 100
                    ward_rows.append({
                        "Ward": w,
                        "Capacity": tot,
                        "Occupied": occ,
                        "Available": av,
                        "Occupancy": f"{pct:.1f}%"
                    })
                st.dataframe(ward_rows, use_container_width=True)
                
        with col_d:
            with st.container(border=True):
                st.markdown('<div class="card-header">🏢 Department Distribution (Active Patients)</div>', unsafe_allow_html=True)
                dept_dist = get_department_distribution()
                if dept_dist:
                    dept_rows = [{"Department": dept, "Active Patients": count} for dept, count in dept_dist.items()]
                    st.dataframe(dept_rows, use_container_width=True)
                else:
                    st.info("No active patients currently assigned to any department.")
                    
        st.write("---")
        
        # Recent admissions & Recent AI Activity side by side
        col_ra, col_ai = st.columns(2)
        with col_ra:
            with st.container(border=True):
                st.markdown('<div class="card-header">📋 Recent Admissions</div>', unsafe_allow_html=True)
                recent_adm = get_recent_admissions(5)
                if recent_adm:
                    adm_rows = [{
                        "ID": p.get("patient_id"),
                        "Name": p.get("name"),
                        "Dept": p.get("department"),
                        "Ward": p.get("ward"),
                        "Bed": p.get("bed_number")
                    } for p in recent_adm]
                    st.dataframe(adm_rows, use_container_width=True)
                else:
                    st.info("No recent active admissions.")
                    
        with col_ai:
            with st.container(border=True):
                st.markdown('<div class="card-header">🤖 Recent AI Activity</div>', unsafe_allow_html=True)
                recent_ai = get_recent_ai_activity(5)
                if recent_ai:
                    ai_rows = [{"User Query": act["query"], "Chat Thread": act["session_title"]} for act in recent_ai]
                    st.dataframe(ai_rows, use_container_width=True)
                else:
                    st.info("No recent AI Assistant queries recorded.")

    # -----------------------------
    # 2. AI ASSISTANT (CHAT)
    # -----------------------------
    elif menu == "🤖 AI Assistant":
        st.title("🤖 VeraOps AI Assistant")
        st.markdown("Modular, LangGraph-backed hospital operations and clinical intelligence.")
        
        # Load sessions on run
        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = load_chat_sessions()
            
        # Layout: Sidebar panel for sessions and main console for chat
        chat_col_left, chat_col_right = st.columns([1.1, 2.9])
        
        with chat_col_left:
            with st.container(border=True):
                st.markdown("### 💬 Chat History")
                if st.button("➕ Start New Chat", use_container_width=True):
                    new_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    st.session_state.chat_sessions[new_id] = {
                        "title": "New Chat Thread",
                        "messages": []
                    }
                    st.session_state.active_session_id = new_id
                    save_chat_sessions(st.session_state.chat_sessions)
                    st.rerun()
                
                st.write("---")
                
                # List past sessions
                for sess_id, sess_data in list(st.session_state.chat_sessions.items()):
                    is_active = (sess_id == st.session_state.active_session_id)
                    btn_text = f"🟢 {sess_data['title']}" if is_active else f"📄 {sess_data['title']}"
                    
                    sess_btn_col, delete_btn_col = st.columns([4, 1])
                    with sess_btn_col:
                        if st.button(btn_text, key=f"sess_{sess_id}", use_container_width=True):
                            st.session_state.active_session_id = sess_id
                            st.rerun()
                    with delete_btn_col:
                        if st.button("🗑", key=f"del_{sess_id}", use_container_width=True):
                            del st.session_state.chat_sessions[sess_id]
                            save_chat_sessions(st.session_state.chat_sessions)
                            
                            if sess_id == st.session_state.active_session_id:
                                if st.session_state.chat_sessions:
                                    st.session_state.active_session_id = list(st.session_state.chat_sessions.keys())[0]
                                else:
                                    new_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    st.session_state.chat_sessions[new_id] = {
                                        "title": "New Chat Thread",
                                        "messages": []
                                    }
                                    st.session_state.active_session_id = new_id
                                    save_chat_sessions(st.session_state.chat_sessions)
                            st.rerun()
                            
        with chat_col_right:
            active_id = st.session_state.active_session_id
            active_sess = st.session_state.chat_sessions.get(active_id, {"title": "New Chat Thread", "messages": []})
            messages = active_sess["messages"]
            
            # Clear chat button
            ccol1, ccol2 = st.columns([6, 1])
            with ccol2:
                if st.button("Clear Conversation", use_container_width=True):
                    active_sess["messages"] = []
                    save_chat_sessions(st.session_state.chat_sessions)
                    st.rerun()
            
            # Predefined suggested prompt buttons
            st.write("💡 **Suggested Queries:**")
            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                if st.button("📋 Summarize patient P001", use_container_width=True):
                    st.session_state.chat_input_val = "Summarize patient P001"
            with scol2:
                if st.button("🔍 Find similar diabetes cases", use_container_width=True):
                    st.session_state.chat_input_val = "Find similar diabetes cases"
            with scol3:
                if st.button("🏥 ICU admission protocol?", use_container_width=True):
                    st.session_state.chat_input_val = "What is the ICU admission protocol?"
                    
            scol4, scol5 = st.columns(2)
            with scol4:
                if st.button("💊 What medications are for patient P002?", use_container_width=True):
                    st.session_state.chat_input_val = "What medications are prescribed for patient P002?"
            with scol5:
                if st.button("🛏 What beds are available?", use_container_width=True):
                    st.session_state.chat_input_val = "What beds are available?"

            st.write("---")
            
            # Display active session messages
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("timestamp"):
                        st.caption(f"🕒 {msg['timestamp']}")
                    if msg["role"] == "assistant" and "metadata" in msg:
                        meta = msg["metadata"]
                        with st.expander("🛠 Agent Reasoning & Explainability Sources"):
                            st.write(f"⏱ **Response Time**: {meta['execution_time']:.2f} seconds")
                            if meta.get("tool_runs"):
                                st.markdown("**Tools Executed:**")
                                for t in meta["tool_runs"]:
                                    st.code(f"Tool: {t['tool']} | Args: {json.dumps(t['args'])}")
                            if meta.get("sources"):
                                st.markdown("**Sources Used / Cited:**")
                                for s in meta["sources"]:
                                    st.write(f"- {s}")
                        
                        # How this answer was generated panel
                        if meta.get("explainability"):
                            with st.expander("🔍 How this answer was generated", expanded=False):
                                exp = meta["explainability"]
                                st.write("**LangGraph Tools Used:**")
                                if exp.get("tools_used"):
                                    st.write(", ".join([f"`{t}`" for t in exp["tools_used"]]))
                                else:
                                    st.write("None")
                                st.write("**Conversation Memory Used:**", exp.get("memory_used", "No"))
                                st.write(f"**Retrieved Chunks Count:** {exp.get('retrieved_chunks_count', 0)}")
                                
                                st.write("**Hospital Documents Retrieved:**")
                                if exp.get("hospital_docs"):
                                    st.write(", ".join([f"`{d}`" for d in exp["hospital_docs"]]))
                                else:
                                    st.write("None")
                                
                                st.write("**Patient Documents Retrieved:**")
                                if exp.get("patient_docs"):
                                    st.write(", ".join([f"`{d}`" for d in exp["patient_docs"]]))
                                else:
                                    st.write("None")
                                    
                                if exp.get("doc_details"):
                                    st.markdown("**Matched Documents Details & FAISS Distances:**")
                                    st.table(exp["doc_details"])
                        
                        # Copy Response expander
                        with st.expander("📋 Copy Response", expanded=False):
                            st.code(msg["content"], language="markdown")
                                    
            # Set prompt value if suggest button clicked
            query = st.chat_input("Enter your clinical or operational question:")
            if "chat_input_val" in st.session_state:
                query = st.session_state.chat_input_val
                del st.session_state.chat_input_val

            if query:
                with st.chat_message("user"):
                    st.markdown(query)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.caption(f"🕒 {ts}")
                    
                if not api_key:
                    st.error("Please add your Groq API Key in the Settings page to run queries.")
                else:
                    with st.spinner("VeraOps agent is reasoning, selecting tools, and processing guidelines..."):
                        try:
                            result = run_agent(query, messages, api_key)
                            ts_assistant = datetime.now().strftime("%Y-%m-%d %H:%M")
                            
                            with st.chat_message("assistant"):
                                st.markdown(result["final_response"])
                                st.caption(f"🕒 {ts_assistant}")
                                
                                # Render Explainability immediately
                                if result.get("explainability"):
                                    with st.expander("🔍 How this answer was generated", expanded=False):
                                        exp = result["explainability"]
                                        st.write("**LangGraph Tools Used:**")
                                        if exp.get("tools_used"):
                                            st.write(", ".join([f"`{t}`" for t in exp["tools_used"]]))
                                        else:
                                            st.write("None")
                                        st.write("**Conversation Memory Used:**", exp.get("memory_used", "No"))
                                        st.write(f"**Retrieved Chunks Count:** {exp.get('retrieved_chunks_count', 0)}")
                                        
                                        st.write("**Hospital Documents Retrieved:**")
                                        if exp.get("hospital_docs"):
                                            st.write(", ".join([f"`{d}`" for d in exp["hospital_docs"]]))
                                        else:
                                            st.write("None")
                                        
                                        st.write("**Patient Documents Retrieved:**")
                                        if exp.get("patient_docs"):
                                            st.write(", ".join([f"`{d}`" for d in exp["patient_docs"]]))
                                        else:
                                            st.write("None")
                                            
                                        if exp.get("doc_details"):
                                            st.markdown("**Matched Documents Details & FAISS Distances:**")
                                            st.table(exp["doc_details"])
                                
                                with st.expander("📋 Copy Response", expanded=False):
                                    st.code(result["final_response"], language="markdown")
                                    
                            # Update active session title if default
                            if active_sess["title"] == "New Chat Thread" or len(messages) == 0:
                                title_text = query[:25] + ("..." if len(query) > 25 else "")
                                active_sess["title"] = title_text
                                
                            messages.append({
                                "role": "user",
                                "content": query,
                                "timestamp": ts
                            })
                            messages.append({
                                "role": "assistant",
                                "content": result["final_response"],
                                "timestamp": ts_assistant,
                                "metadata": {
                                    "execution_time": result["execution_time"],
                                    "tool_runs": result["tool_runs"],
                                    "sources": result["sources"],
                                    "explainability": result["explainability"]
                                }
                            })
                            
                            save_chat_sessions(st.session_state.chat_sessions)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error executing query: {e}")

    # -----------------------------
    # 3. PATIENT REGISTRATION
    # -----------------------------
    elif menu == "👤 Patient Registration":
        st.title("👤 Patient Admission Registration")
        st.markdown("Organized clinical admission pathway for registering new patient records.")
        
        next_id = get_next_patient_id()
        
        with st.form("redesigned_registration_form"):
            st.markdown("### 📋 SECTION 1: Personal Information")
            st.text_input("Patient ID (Auto Generated)", value=next_id, disabled=True)
            
            r_name = st.text_input("Full Name *", placeholder="Enter patient's full name")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                r_age = st.number_input("Age *", min_value=0, max_value=130, value=35, step=1)
            with col_p2:
                r_gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
            with col_p3:
                r_blood = st.selectbox("Blood Group *", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"])
                
            col_p4, col_p5 = st.columns(2)
            with col_p4:
                r_phone = st.text_input("Phone Number", placeholder="+1-555-XXX-XXXX")
            with col_p5:
                r_email = st.text_input("Email Address", placeholder="patient@example.com")
                
            r_addr = st.text_area("Address", placeholder="Enter residential address")
            
            col_p6, col_p7 = st.columns(2)
            with col_p6:
                r_emergency_name = st.text_input("Emergency Contact Name")
            with col_p7:
                r_emergency_phone = st.text_input("Emergency Contact Number")
                
            st.markdown("### 🚪 SECTION 2: Admission Information")
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                r_date = st.text_input("Date of Admission", value=datetime.now().strftime("%Y-%m-%d"))
            with col_a2:
                r_dept = st.selectbox("Department", ["Cardiology", "Neurology", "Orthopedics", "Emergency", "ICU", "General Medicine", "Pulmonology", "Nephrology", "Pediatrics", "Surgery"])
            with col_a3:
                r_doctor = st.text_input("Assigned Doctor", value="Dr. Vance, MD")
                
            col_a4, col_a5 = st.columns(2)
            with col_a4:
                r_ward = st.selectbox("Ward", ["Cardiology", "Neurology", "Orthopedics", "Emergency", "ICU", "General Medicine", "Pulmonology", "Nephrology", "Pediatrics", "Surgery"])
            with col_a5:
                # Load available beds dynamically
                avail_beds = get_available_beds(r_ward)
                if avail_beds:
                    r_bed = st.selectbox("Bed Number", avail_beds)
                else:
                    st.error("No available beds in this ward! Please select another ward.")
                    r_bed = None
                    
            col_a6, col_a7 = st.columns(2)
            with col_a6:
                r_adm_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Referral"])
            with col_a7:
                r_status = st.selectbox("Current Status", ["Admitted", "Observation", "ICU"])
                
            st.markdown("### 🏥 SECTION 3: Clinical Information")
            r_complaint = st.text_area("Chief Complaint *", placeholder="What is the primary reason for admission?")
            r_diagnosis = st.text_input("Primary Diagnosis", placeholder="Initial working diagnosis")
            r_symptoms = st.text_area("Symptoms", placeholder="Describe observed clinical symptoms")
            r_history = st.text_area("Past Medical History", value="None")
            r_allergies = st.text_input("Known Allergies", value="NKDA")
            r_meds = st.text_area("Current Medications", value="None")
            
            st.markdown("### ⚙️ SECTION 4: Optional Information")
            col_o1, col_o2, col_o3 = st.columns(3)
            with col_o1:
                r_height = st.number_input("Height (cm)", min_value=0.0, max_value=300.0, value=170.0, step=0.5)
            with col_o2:
                r_weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=70.0, step=0.1)
            with col_o3:
                r_bmi = 0.0
                if r_height > 0:
                    r_bmi = r_weight / ((r_height / 100.0) ** 2)
                st.text_input("BMI (Auto Calculated)", value=f"{r_bmi:.2f}", disabled=True)
                
            col_o4, col_o5, col_o6 = st.columns(3)
            with col_o4:
                r_ins_prov = st.text_input("Insurance Provider", value="N/A")
            with col_o5:
                r_ins_num = st.text_input("Insurance Number", value="N/A")
            with col_o6:
                r_national_id = st.text_input("National ID / Hospital ID", value="N/A")
                
            st.markdown("*(Fields marked with * are required)*")
            submit_reg = st.form_submit_button("🏥 REGISTER PATIENT")
            
            if submit_reg:
                if not r_name.strip() or not r_complaint.strip():
                    st.error("Patient Name and Chief Complaint are required.")
                elif not r_bed:
                    st.error("Cannot register patient: No available bed selected.")
                else:
                    reg_data = {
                        "patient_id": next_id,
                        "name": r_name.strip(),
                        "age": int(r_age),
                        "gender": r_gender,
                        "blood_group": r_blood,
                        "phone_number": r_phone.strip(),
                        "email": r_email.strip(),
                        "address": r_addr.strip(),
                        "emergency_contact_name": r_emergency_name.strip(),
                        "emergency_contact_number": r_emergency_phone.strip(),
                        "date_of_admission": r_date.strip(),
                        "department": r_dept,
                        "assigned_doctor": r_doctor.strip(),
                        "ward": r_ward,
                        "bed_number": r_bed,
                        "admission_type": r_adm_type,
                        "current_status": r_status,
                        "chief_complaint": r_complaint.strip(),
                        "diagnosis": r_diagnosis.strip(),
                        "symptoms": r_symptoms.strip(),
                        "past_medical_history": r_history.strip(),
                        "allergies": r_allergies.strip(),
                        "current_medications": r_meds.strip(),
                        "height": r_height,
                        "weight": r_weight,
                        "bmi": round(r_bmi, 2),
                        "insurance_provider": r_ins_prov.strip(),
                        "insurance_number": r_ins_num.strip(),
                        "national_id": r_national_id.strip()
                    }
                    
                    with st.spinner("Registering patient, allocating bed, generating templates, and updating RAG vector index..."):
                        try:
                            msg = register_new_patient(reg_data)
                            st.success(msg)
                            st.session_state.selected_patient_id = next_id
                            st.session_state.current_page = "🩺 Doctor Workspace"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to register patient: {e}")

    # -----------------------------
    # 4. DOCTOR WORKSPACE
    # -----------------------------
    elif menu == "🩺 Doctor Workspace":
        st.title("🩺 Doctor Workspace")
        st.markdown("Clinical profiling, progression logging, and AI case summaries.")
        
        patients = get_all_patients()
        if not patients:
            st.info("No patients available in database.")
        else:
            patient_options = {f"{p['patient_id']} - {p['name']}": p['patient_id'] for p in patients}
            
            # Autocomplete selectbox index helper
            selected_idx = 0
            if "selected_patient_id" in st.session_state:
                target_id = st.session_state.selected_patient_id
                for idx, (label, pid) in enumerate(patient_options.items()):
                    if pid == target_id:
                        selected_idx = idx
                        break
            
            selected_pt_label = st.selectbox("🔍 Search & Select Patient Profile", list(patient_options.keys()), index=selected_idx)
            selected_pt_id = patient_options[selected_pt_label]
            st.session_state.selected_patient_id = selected_pt_id
            
            p_record = get_patient(selected_pt_id)
            if p_record:
                p_dict = dict(p_record)
                
                # Tabs
                w_tab1, w_tab2, w_tab3, w_tab4, w_tab5, w_tab6, w_tab7, w_tab8, w_tab9 = st.tabs([
                    "📋 Patient Summary",
                    "📝 Doctor Notes",
                    "💊 Prescription",
                    "🔬 Laboratory",
                    "🩻 Radiology",
                    "📋 Treatment Plan",
                    "🚪 Discharge",
                    "🤖 AI Clinical Summary",
                    "📈 Patient Timeline"
                ])
                
                # Tab 1: Patient Summary
                with w_tab1:
                    st.subheader("Patient Clinical Profile Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Personal Details")
                        st.write(f"**Patient ID**: {p_dict.get('patient_id')}")
                        st.write(f"**Full Name**: {p_dict.get('name')}")
                        st.write(f"**Age**: {p_dict.get('age')}")
                        st.write(f"**Gender**: {p_dict.get('gender')}")
                        st.write(f"**Blood Group**: {p_dict.get('blood_group')}")
                        st.write(f"**Phone Number**: {p_dict.get('phone_number')}")
                        st.write(f"**Email**: {p_dict.get('email')}")
                        st.write(f"**Address**: {p_dict.get('address')}")
                        st.write(f"**Emergency Contact**: {p_dict.get('emergency_contact_name')} ({p_dict.get('emergency_contact_number')})")
                    
                    with col2:
                        st.write("### Administrative & Measurements")
                        st.write(f"**Date of Admission**: {p_dict.get('date_of_admission')}")
                        st.write(f"**Department**: {p_dict.get('department')}")
                        st.write(f"**Assigned Doctor**: {p_dict.get('assigned_doctor')}")
                        st.write(f"**Ward**: {p_dict.get('ward') or 'Discharged'}")
                        st.write(f"**Bed**: {p_dict.get('bed_number') or 'N/A'}")
                        st.write(f"**Admission Type**: {p_dict.get('admission_type')}")
                        st.write(f"**Current Status**: {p_dict.get('current_status')}")
                        st.write(f"**Height**: {p_dict.get('height')} cm")
                        st.write(f"**Weight**: {p_dict.get('weight')} kg")
                        st.write(f"**BMI**: {p_dict.get('bmi')}")
                        st.write(f"**Insurance Provider**: {p_dict.get('insurance_provider')}")
                        st.write(f"**Insurance Number**: {p_dict.get('insurance_number')}")
                        st.write(f"**National ID**: {p_dict.get('national_id')}")
                        
                    st.write("---")
                    st.write("### Clinical Presentation")
                    st.write(f"**Chief Complaint**: {p_dict.get('chief_complaint')}")
                    st.write(f"**Primary Diagnosis**: {p_dict.get('diagnosis')}")
                    st.write(f"**Symptoms**: {p_dict.get('symptoms')}")
                    st.write(f"**Past Medical History**: {p_dict.get('past_medical_history')}")
                    st.write(f"**Known Allergies**: {p_dict.get('allergies')}")
                    st.write(f"**Current Medications**: {p_dict.get('current_medications')}")

                # Tab 2: Doctor Notes
                with w_tab2:
                    st.subheader("Daily Clinical Notes Log")
                    
                    notes_file = os.path.join(PROJECT_ROOT, "patient_documents", selected_pt_id, "doctor_notes.md")
                    current_notes_content = ""
                    if os.path.exists(notes_file):
                        try:
                            with open(notes_file, "r", encoding="utf-8") as f:
                                current_notes_content = f.read()
                        except Exception:
                            pass
                    
                    st.markdown("**Current Clinical Notes on File:**")
                    if current_notes_content:
                        st.info(current_notes_content)
                    else:
                        st.write("*No notes recorded yet.*")
                        
                    st.write("---")
                    new_note = st.text_area("Add New Clinical Progress Note", height=150)
                    if st.button("💾 Save Note", key="save_notes_btn"):
                        if new_note.strip():
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                            appended_content = f"\n\n### Clinical Progress Note — {timestamp}\n{new_note.strip()}"
                            try:
                                with open(notes_file, "a", encoding="utf-8") as f:
                                    f.write(appended_content)
                                updated_notes_field = (p_dict.get("visit_notes") or "") + appended_content
                                update_patient(selected_pt_id, {"visit_notes": updated_notes_field})
                                add_timeline_event(selected_pt_id, "Doctor Notes Added", f"Appended clinical progress note: '{new_note.strip()[:100]}...'", doctor=p_dict.get("assigned_doctor"))
                                index_patient_documents(selected_pt_id)
                                st.success("Doctor note appended and patient RAG index refreshed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to save note: {e}")
                        else:
                            st.warning("Note content cannot be empty.")

                # Tab 3: Prescription
                with w_tab3:
                    st.subheader("Prescribe Outpatient Medications")
                    
                    presc_file = os.path.join(PROJECT_ROOT, "patient_documents", selected_pt_id, "prescription.md")
                    current_rx_content = ""
                    if os.path.exists(presc_file):
                        try:
                            with open(presc_file, "r", encoding="utf-8") as f:
                                current_rx_content = f.read()
                        except Exception:
                            pass
                            
                    st.markdown("**Current Active Prescription:**")
                    if current_rx_content:
                        st.info(current_rx_content)
                    else:
                        st.write("*No prescriptions recorded yet.*")
                        
                    st.write("---")
                    st.write("#### Add / Update Prescription Details")
                    
                    rx_med = st.text_input("Medicine Name")
                    rx_dosage = st.text_input("Dosage")
                    rx_freq = st.text_input("Frequency")
                    rx_dur = st.text_input("Duration")
                    rx_instr = st.text_area("Special Instructions")
                    
                    if st.button("💾 Add Medication to Prescription", key="save_rx_btn"):
                        if rx_med.strip():
                            new_row = f"\n| {rx_med.strip()} | {rx_dosage.strip()} | {rx_freq.strip()} | {rx_dur.strip()} | {rx_instr.strip()} |"
                            if not current_rx_content.strip() or "| Medicine Name |" not in current_rx_content:
                                header = (
                                    f"# Medical Prescription\n\n"
                                    f"- **Patient ID**: {selected_pt_id}\n"
                                    f"- **Patient Name**: {p_dict.get('name')}\n"
                                    f"- **Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
                                    f"- **Prescribing Doctor**: {p_dict.get('assigned_doctor')}\n\n"
                                    f"## Known Allergies\n{p_dict.get('allergies', 'NKDA')}\n\n"
                                    f"## Active Medications List\n\n"
                                    f"| Medicine Name | Dosage | Frequency | Duration | Route / Instructions |\n"
                                    f"| :--- | :--- | :--- | :--- | :--- |\n"
                                )
                                new_rx_content = header + f"| {rx_med.strip()} | {rx_dosage.strip()} | {rx_freq.strip()} | {rx_dur.strip()} | {rx_instr.strip()} |"
                            else:
                                new_rx_content = current_rx_content.strip() + new_row
                                
                            try:
                                with open(presc_file, "w", encoding="utf-8") as f:
                                    f.write(new_rx_content)
                                    
                                medicines_field = p_dict.get("medicines", "")
                                if medicines_field:
                                    medicines_field += f"; {rx_med.strip()} {rx_dosage.strip()} {rx_freq.strip()}"
                                else:
                                    medicines_field = f"{rx_med.strip()} {rx_dosage.strip()} {rx_freq.strip()}"
                                    
                                update_patient(selected_pt_id, {"medicines": medicines_field})
                                add_timeline_event(selected_pt_id, "Prescription Updated", f"Added medication: {rx_med.strip()} {rx_dosage.strip()} {rx_freq.strip()}.", doctor=p_dict.get("assigned_doctor"))
                                index_patient_documents(selected_pt_id)
                                st.success("Prescription updated successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to update prescription: {e}")
                        else:
                            st.warning("Medicine Name is required.")

                # Tab 4: Laboratory
                with w_tab4:
                    st.subheader("Laboratory Investigations")
                    
                    lab_file = os.path.join(PROJECT_ROOT, "patient_documents", selected_pt_id, "lab_report.md")
                    current_lab_content = ""
                    if os.path.exists(lab_file):
                        try:
                            with open(lab_file, "r", encoding="utf-8") as f:
                                current_lab_content = f.read()
                        except Exception:
                            pass
                            
                    st.markdown("**Current Lab Reports on File:**")
                    if current_lab_content:
                        st.info(current_lab_content)
                    else:
                        st.write("*No lab reports recorded.*")
                        
                    st.write("---")
                    lab_input_type = st.radio("Add Method", ["Enter Findings Manually", "Upload Report File"])
                    
                    if lab_input_type == "Enter Findings Manually":
                        lab_findings = st.text_area("Enter lab findings details (CBC, biochemistry, electrolytes, etc.)", height=150)
                        if st.button("💾 Save Lab Investigation", key="save_lab_btn"):
                            if lab_findings.strip():
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                                new_content = f"\n\n## Lab Findings — {timestamp}\n{lab_findings.strip()}"
                                try:
                                    with open(lab_file, "a", encoding="utf-8") as f:
                                        f.write(new_content)
                                    add_timeline_event(selected_pt_id, "Lab Updates", f"Updated laboratory findings report manually: '{lab_findings.strip()[:100]}...'", doctor=p_dict.get("assigned_doctor"))
                                    index_patient_documents(selected_pt_id)
                                    st.success("Lab report updated!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to save lab report: {e}")
                            else:
                                st.warning("Findings cannot be empty.")
                    else:
                        lab_upload = st.file_uploader("Upload Lab Report (Markdown format)", type=["md"])
                        if lab_upload is not None:
                            if st.button("💾 Save Uploaded Lab Report", key="save_uploaded_lab"):
                                try:
                                    uploaded_txt = lab_upload.read().decode("utf-8")
                                    with open(lab_file, "w", encoding="utf-8") as f:
                                        f.write(uploaded_txt)
                                    add_timeline_event(selected_pt_id, "Lab Updates", f"Uploaded laboratory findings report file: {lab_upload.name}.", doctor=p_dict.get("assigned_doctor"))
                                    index_patient_documents(selected_pt_id)
                                    st.success("Uploaded lab report saved and indexed!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to save uploaded lab report: {e}")

                # Tab 5: Radiology
                with w_tab5:
                    st.subheader("Radiology & Imaging Reports")
                    
                    rad_file = os.path.join(PROJECT_ROOT, "patient_documents", selected_pt_id, "radiology_report.md")
                    current_rad_content = ""
                    if os.path.exists(rad_file):
                        try:
                            with open(rad_file, "r", encoding="utf-8") as f:
                                current_rad_content = f.read()
                        except Exception:
                            pass
                            
                    st.markdown("**Current Radiology Investigations:**")
                    if current_rad_content:
                        st.info(current_rad_content)
                    else:
                        st.write("*No radiology reports recorded.*")
                        
                    st.write("---")
                    rad_input_type = st.radio("Add Method", ["Enter Findings Manually", "Upload Report File"], key="rad_input_type")
                    
                    if rad_input_type == "Enter Findings Manually":
                        rad_findings = st.text_area("Enter radiology findings (CT, MRI, X-Ray, Echo, etc.)", height=150)
                        if st.button("💾 Save Radiology Investigation", key="save_rad_btn"):
                            if rad_findings.strip():
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                                new_content = f"\n\n## Radiology Findings — {timestamp}\n{rad_findings.strip()}"
                                try:
                                    with open(rad_file, "a", encoding="utf-8") as f:
                                        f.write(new_content)
                                    add_timeline_event(selected_pt_id, "Radiology Updates", f"Updated radiology findings report manually: '{rad_findings.strip()[:100]}...'", doctor=p_dict.get("assigned_doctor"))
                                    index_patient_documents(selected_pt_id)
                                    st.success("Radiology report updated!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to save radiology report: {e}")
                            else:
                                st.warning("Findings cannot be empty.")
                    else:
                        rad_upload = st.file_uploader("Upload Radiology Report (Markdown format)", type=["md"], key="rad_uploader")
                        if rad_upload is not None:
                            if st.button("💾 Save Uploaded Radiology Report", key="save_uploaded_rad"):
                                try:
                                    uploaded_txt = rad_upload.read().decode("utf-8")
                                    with open(rad_file, "w", encoding="utf-8") as f:
                                        f.write(uploaded_txt)
                                    add_timeline_event(selected_pt_id, "Radiology Updates", f"Uploaded radiology findings report file: {rad_upload.name}.", doctor=p_dict.get("assigned_doctor"))
                                    index_patient_documents(selected_pt_id)
                                    st.success("Uploaded radiology report saved and indexed!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to save uploaded radiology report: {e}")

                # Tab 6: Treatment Plan
                with w_tab6:
                    st.subheader("Clinical Treatment Guidelines & Progression")
                    
                    tx_file = os.path.join(PROJECT_ROOT, "patient_documents", selected_pt_id, "treatment_plan.md")
                    current_tx_content = ""
                    if os.path.exists(tx_file):
                        try:
                            with open(tx_file, "r", encoding="utf-8") as f:
                                current_tx_content = f.read()
                        except Exception:
                            pass
                            
                    st.markdown("**Current Treatment Plan on File:**")
                    if current_tx_content:
                        st.info(current_tx_content)
                    else:
                        st.write("*No treatment plan recorded.*")
                        
                    st.write("---")
                    st.write("#### Add Updates to Treatment Plan")
                    tx_treatment = st.text_area("Current Treatment & Medication Changes")
                    tx_progress = st.text_area("Clinical Progress & Observations")
                    
                    if st.button("💾 Save Treatment Plan Update", key="save_tx_plan_btn"):
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                        new_content = (
                            f"\n\n## Treatment Update — {timestamp}\n"
                            f"- **Medication & Treatment Changes**: {tx_treatment.strip() if tx_treatment.strip() else 'N/A'}\n"
                            f"- **Clinical Progress & Observations**: {tx_progress.strip() if tx_progress.strip() else 'N/A'}\n"
                        )
                        try:
                            with open(tx_file, "a", encoding="utf-8") as f:
                                f.write(new_content)
                            add_timeline_event(selected_pt_id, "Treatment Plan Updates", f"Updated treatment plan: {tx_treatment.strip()[:100]}...", doctor=p_dict.get("assigned_doctor"))
                            index_patient_documents(selected_pt_id)
                            st.success("Treatment plan updated and patient vectors refreshed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save treatment plan update: {e}")

                # Tab 7: Discharge Summary
                with w_tab7:
                    st.subheader("Patient Discharge Process")
                    
                    if p_dict.get("current_status") == "Discharged":
                        st.success(f"Patient was already discharged on {p_dict.get('date_of_discharge')}.")
                        dis_summary_file = os.path.join(PROJECT_ROOT, "patient_documents", selected_pt_id, "discharge_summary.md")
                        if os.path.exists(dis_summary_file):
                            with open(dis_summary_file, "r", encoding="utf-8") as f:
                                st.markdown(f.read())
                    else:
                        st.warning("⚠️ Discharging a patient will update their database status, release their hospital bed, and compile a Discharge Summary.")
                        dis_diagnosis = st.text_input("Final Diagnosis", value=p_dict.get("diagnosis", ""))
                        dis_date = st.text_input("Discharge Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
                        dis_meds = st.text_area("Final Discharge Medications / Prescription", value=p_dict.get("medicines", ""))
                        dis_notes = st.text_area("Discharge Summary / Course of Treatment", value="Patient recovered and stabilized under protocol.")
                        dis_follow = st.text_input("Follow-up Date & Instructions", value="Scheduled in 1 week in outpatient department.")
                        
                        if st.button("🚪 Process & Discharge Patient", key="discharge_btn"):
                            dis_data = {
                                "current_status": "Discharged",
                                "diagnosis": dis_diagnosis,
                                "date_of_discharge": dis_date,
                                "discharge_summary": dis_notes,
                                "follow_up_date": dis_follow,
                                "medicines": dis_meds,
                                "ward": "",
                                "bed_number": ""
                            }
                            
                            try:
                                update_patient(selected_pt_id, dis_data)
                                release_bed(selected_pt_id)
                                
                                gen_dis_data = {
                                    "Patient ID": selected_pt_id,
                                    "Full Name": p_dict.get("name"),
                                    "Assigned Doctor": p_dict.get("assigned_doctor"),
                                    "Date of Admission": p_dict.get("date_of_admission"),
                                    "Date of Discharge": dis_date,
                                    "Disease / Diagnosis": dis_diagnosis,
                                    "Discharge Summary": dis_notes,
                                    "Prescription": dis_meds,
                                    "Follow-up Date": dis_follow
                                }
                                
                                dis_path = os.path.join(PROJECT_ROOT, "patient_documents", selected_pt_id, "discharge_summary.md")
                                from scripts.seed_database import generate_discharge_summary
                                with open(dis_path, "w", encoding="utf-8") as f:
                                    f.write(generate_discharge_summary(gen_dis_data))
                                    
                                add_timeline_event(selected_pt_id, "Discharge", f"Patient discharged from ward {p_dict.get('ward') or 'N/A'}. Final Diagnosis: {dis_diagnosis}.", doctor=p_dict.get("assigned_doctor"))
                                index_patient_documents(selected_pt_id)
                                
                                st.success("Patient discharged successfully! Bed released and FAISS index updated.")
                                if "selected_patient_id" in st.session_state:
                                    del st.session_state.selected_patient_id
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to discharge patient: {e}")

                # Tab 8: AI Clinical Summary (Feature 6)
                with w_tab8:
                    st.subheader("Generate AI Clinical Summary")
                    st.markdown("Invokes the supervisor agent to read patient records, doctor notes, lab reports, and radiology summaries to synthesize clinical findings.")
                    
                    if not api_key:
                        st.error("Please add your Groq API Key in the Settings page.")
                    else:
                        if st.button("🤖 Generate Summary Now", key="ai_sum_btn"):
                            with st.spinner("Synthesizing patient history, laboratory diagnostics, and clinical progression..."):
                                ai_query = (
                                    f"Generate a comprehensive AI clinical summary report for patient {selected_pt_id}. "
                                    f"Analyze all documents including: patient_summary.md, doctor_notes.md, prescription.md, treatment_plan.md, lab_report.md, radiology_report.md. "
                                    f"Structure the final output with exactly these markdown headings:\n"
                                    f"## Current Patient Summary\n"
                                    f"## Clinical Progress\n"
                                    f"## Important Findings\n"
                                    f"## Potential Risks\n"
                                    f"## Suggested Follow-up\n"
                                    f"Be clinically precise and cite specific findings where applicable."
                                )
                                try:
                                    res = run_agent(ai_query, [], api_key)
                                    st.session_state.ai_clinical_summary_result = res["final_response"]
                                    st.session_state.ai_summary_generated_for = selected_pt_id
                                except Exception as e:
                                    st.error(f"Failed to execute supervisor: {e}")
                                    
                        if "ai_clinical_summary_result" in st.session_state and st.session_state.get("ai_summary_generated_for") == selected_pt_id:
                            st.write("---")
                            st.write("### AI Synthesized Report")
                            st.markdown(st.session_state.ai_clinical_summary_result)
                            
                            st.write("---")
                            if st.button("💾 Save as ai_clinical_summary.md", key="save_ai_sum_btn"):
                                summary_path = os.path.join(PROJECT_ROOT, "patient_documents", selected_pt_id, "ai_clinical_summary.md")
                                try:
                                    with open(summary_path, "w", encoding="utf-8") as f:
                                        f.write(st.session_state.ai_clinical_summary_result)
                                    add_timeline_event(selected_pt_id, "AI Clinical Summary", "Generated and saved AI clinical summary report.", doctor=p_dict.get("assigned_doctor"))
                                    index_patient_documents(selected_pt_id)
                                    st.success("Saved AI Clinical Summary as 'ai_clinical_summary.md' and refreshed Patient RAG index!")
                                    del st.session_state.ai_clinical_summary_result
                                    del st.session_state.ai_summary_generated_for
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to save AI Clinical Summary: {e}")

                # Tab 9: Patient Timeline
                with w_tab9:
                    st.subheader("Patient Chronological Timeline")
                    timeline_events = get_patient_timeline(selected_pt_id)
                    if not timeline_events:
                        st.info("No timeline events logged for this patient.")
                    else:
                        st.markdown("Chronological list of all medical and administrative events:")
                        for event in timeline_events:
                            with st.container(border=True):
                                t_col1, t_col2 = st.columns([3, 1])
                                with t_col1:
                                    st.markdown(f"#### 🏷️ Event: {event['event_type']}")
                                with t_col2:
                                    st.caption(f"🕒 {event['event_timestamp']}")
                                    
                                if event.get("doctor") and event["doctor"] != "None":
                                    st.write(f"🧑‍⚕️ **Responsible**: {event['doctor']}")
                                st.write(f"📝 **Description**: {event['description']}")

    # -----------------------------
    # 5. PATIENT DIRECTORY
    # -----------------------------
    elif menu == "📋 Patient Directory":
        st.title("📋 Patient Directory & Case Search")
        
        patients = get_all_patients()
        
        dir_tab1, dir_tab2 = st.tabs(["📋 Patient Master Directory", "🔍 Similar Patient Search"])
        
        with dir_tab1:
            st.markdown("Searchable, sortable, and filtered master directory of all registered hospital records.")
            if not patients:
                st.info("No patient records exist in SQLite database.")
            else:
                with st.expander("🔍 Filter & Sort Patients", expanded=True):
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        global_search = st.text_input("Global Search (ID, Name, Diagnosis)", key="dir_search")
                        gender_options = ["All"] + sorted(list({p["gender"] for p in patients if p.get("gender")}))
                        sel_gender = st.selectbox("Gender", gender_options)
                        
                        ward_options = ["All"] + [
                            "Cardiology", "Neurology", "Orthopedics", "Emergency", "ICU",
                            "General Medicine", "Pulmonology", "Nephrology", "Pediatrics", "Surgery"
                        ]
                        sel_ward = st.selectbox("Ward", ward_options)
                        
                    with col_f2:
                        dept_options = ["All"] + sorted(list({p["department"] for p in patients if p.get("department")}))
                        sel_dept = st.selectbox("Department", dept_options)
                        
                        doc_options = ["All"] + sorted(list({p["assigned_doctor"] for p in patients if p.get("assigned_doctor")}))
                        sel_doc = st.selectbox("Assigned Doctor", doc_options)
                        
                        status_options = ["All"] + sorted(list({p["current_status"] for p in patients if p.get("current_status")}))
                        sel_status = st.selectbox("Current Status", status_options)
                        
                    with col_f3:
                        ages = [int(p["age"]) for p in patients if p.get("age") is not None]
                        min_age = min(ages) if ages else 0
                        max_age = max(ages) if ages else 120
                        sel_age_range = st.slider("Age Range", int(min_age), int(max_age), (int(min_age), int(max_age)))
                        sel_adm_date = st.text_input("Admission Date (YYYY-MM-DD)", placeholder="All")
                        
                        sort_by = st.selectbox("Sort By", ["Patient ID", "Name", "Age", "Admission Date"])
                        sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)

                # Apply Filtering
                filtered_patients = []
                for p in patients:
                    if global_search:
                        q = global_search.lower()
                        id_match = q in str(p.get("patient_id", "")).lower()
                        name_match = q in str(p.get("name", "")).lower()
                        diag_match = q in str(p.get("diagnosis", "")).lower()
                        if not (id_match or name_match or diag_match):
                            continue
                            
                    if sel_gender != "All" and p.get("gender") != sel_gender:
                        continue
                    if sel_dept != "All" and p.get("department") != sel_dept:
                        continue
                    if sel_doc != "All" and p.get("assigned_doctor") != sel_doc:
                        continue
                    if sel_ward != "All" and p.get("ward") != sel_ward:
                        continue
                    if sel_status != "All" and p.get("current_status") != sel_status:
                        continue
                    age_val = p.get("age")
                    if age_val is not None:
                        if not (sel_age_range[0] <= int(age_val) <= sel_age_range[1]):
                            continue
                    if sel_adm_date and sel_adm_date.strip():
                        if sel_adm_date.strip() not in str(p.get("date_of_admission", "")):
                            continue
                            
                    filtered_patients.append(p)
                    
                # Apply Sorting
                key_map = {
                    "Patient ID": "patient_id",
                    "Name": "name",
                    "Age": "age",
                    "Admission Date": "date_of_admission"
                }
                sort_key = key_map.get(sort_by, "patient_id")
                
                if sort_key == "age":
                    filtered_patients.sort(key=lambda x: int(x.get("age") or 0), reverse=(sort_order == "Descending"))
                else:
                    filtered_patients.sort(key=lambda x: str(x.get(sort_key) or ""), reverse=(sort_order == "Descending"))
                    
                # Pagination
                st.markdown(f"**Found {len(filtered_patients)} patients**")
                page_size = st.selectbox("Rows per page", [10, 25, 50], index=0)
                total_pages = (len(filtered_patients) + page_size - 1) // page_size
                
                if total_pages > 1:
                    current_page = st.number_input("Page Selector", min_value=1, max_value=total_pages, value=1)
                else:
                    current_page = 1
                    
                start_idx = (current_page - 1) * page_size
                end_idx = min(start_idx + page_size, len(filtered_patients))
                page_pts = filtered_patients[start_idx:end_idx]
                
                if page_pts:
                    table_data = []
                    for p in page_pts:
                        table_data.append({
                            "Patient ID": p.get("patient_id"),
                            "Name": p.get("name"),
                            "Age": p.get("age"),
                            "Gender": p.get("gender"),
                            "Department": p.get("department"),
                            "Assigned Doctor": p.get("assigned_doctor"),
                            "Ward": p.get("ward") or "Discharged",
                            "Bed": p.get("bed_number") or "N/A",
                            "Diagnosis": p.get("diagnosis"),
                            "Status": p.get("current_status"),
                            "Admission Date": p.get("date_of_admission"),
                            "Discharge Date": p.get("date_of_discharge") or "N/A"
                        })
                    st.dataframe(table_data, use_container_width=True)
                    
                    st.write("---")
                    st.subheader("Profile Selection")
                    target_options = {f"{p['patient_id']} - {p['name']}": p['patient_id'] for p in page_pts}
                    selected_open_id = st.selectbox("Select Patient to Open Profile", list(target_options.keys()))
                    actual_id = target_options[selected_open_id]
                    
                    if st.button("🩺 Open Patient Profile in Doctor Workspace", use_container_width=True):
                        st.session_state.selected_patient_id = actual_id
                        st.session_state.current_page = "🩺 Doctor Workspace"
                        st.rerun()
                else:
                    st.info("No records match your criteria.")
                    
        with dir_tab2:
            st.subheader("Semantic Similar Case Search")
            st.markdown("Query the FAISS vector database to retrieve historical cases with matching symptoms or clinical history, alongside their SQLite database records.")
            
            sim_query = st.text_input("Enter clinical symptoms or diagnosis query (e.g. 'fever with productive cough and chest congestion')", placeholder="Search term...", key="sim_query_inp")
            sim_k = st.slider("Number of cases to retrieve", min_value=1, max_value=10, value=3, key="sim_k_slider")
            
            if st.button("🔍 Search Similar Patients", key="search_similar_btn"):
                if not sim_query.strip():
                    st.warning("Please enter a clinical query.")
                else:
                    with st.spinner("Searching FAISS patient vectors..."):
                        try:
                            matches = search_similar_patients(sim_query.strip(), k=sim_k)
                            
                            if not matches:
                                st.info("No matching historical cases found.")
                            else:
                                sim_rows = []
                                for m in matches:
                                    pid = m["patient_id"]
                                    dist = m["distance"]
                                    # Similarity score conversion:
                                    score_pct = max(0.0, min(100.0, (1.0 - (dist / 2.0)) * 100.0))
                                    
                                    # Fetch details from SQLite
                                    p_record = get_patient(pid)
                                    if p_record:
                                        sim_rows.append({
                                            "Patient ID": pid,
                                            "Name": p_record["name"],
                                            "Diagnosis": p_record["diagnosis"],
                                            "Department": p_record["department"],
                                            "Current Status": p_record["current_status"],
                                            "Admission Date": p_record["date_of_admission"],
                                            "Similarity Score": f"{score_pct:.1f}%"
                                        })
                                        
                                if sim_rows:
                                    st.dataframe(sim_rows, use_container_width=True)
                                    
                                    st.write("---")
                                    st.subheader("Redirection Options")
                                    # Open matched profile
                                    target_pids = {f"{r['Patient ID']} - {r['Name']}": r['Patient ID'] for r in sim_rows}
                                    selected_redir = st.selectbox("Select Patient to view profile in workspace", list(target_pids.keys()), key="sim_redir_select")
                                    actual_redir_id = target_pids[selected_redir]
                                    
                                    if st.button("🩺 Open Patient Profile", key="sim_redir_btn"):
                                        st.session_state.selected_patient_id = actual_redir_id
                                        st.session_state.current_page = "🩺 Doctor Workspace"
                                        st.rerun()
                                else:
                                    st.info("No structured database records found for matches.")
                        except Exception as e:
                            st.error(f"Failed to retrieve similar cases: {e}")

    # -----------------------------
    # 6. KNOWLEDGE BASE
    # -----------------------------
    elif menu == "📚 Knowledge Base":
        st.title("📚 VeraOps Knowledge Base")
        st.markdown("Manage vector search indexes, hospital protocols, and clinical document directories.")
        
        stats = get_knowledge_base_stats()
        
        # Overview Layout
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown('<div class="card-header">Hospital Guidelines Index</div>', unsafe_allow_html=True)
                st.write(f"- SOP / Protocol Files: **{stats['hospital_docs_count']}**")
                st.write(f"- Indexed Guidelines Chunks: **{stats['hospital_vector_count']}**")
            
        with col2:
            with st.container(border=True):
                st.markdown('<div class="card-header">Patient Health Record Index</div>', unsafe_allow_html=True)
                st.write(f"- Patient Documents Files: **{stats['patient_docs_count']}**")
                st.write(f"- Indexed Patients Chunks: **{stats['patient_vector_count']}**")
            
        st.write("---")
        
        st.subheader("Indexing Metrics")
        kcol1, kcol2, kcol3 = st.columns(3)
        with kcol1:
            st.metric("Total Indexed Chunks", stats["total_chunks"])
        with kcol2:
            st.metric("Embedding Model", stats["embedding_model"])
        with kcol3:
            if check_needs_rebuild():
                st.metric("Index Status", "Rebuild Needed")
            else:
                st.metric("Index Status", "Up to Date")
                
        st.write("---")
        
        # Document uploaders
        st.subheader("Upload SOPs / Protocol Files")
        upcol1, upcol2 = st.columns(2)
        with upcol1:
            st.write("📂 **Upload Hospital SOP (PDF/MD)**")
            uploaded_h_file = st.file_uploader("Upload to hospital_docs/", type=["pdf", "md"])
            if uploaded_h_file is not None:
                save_dir = os.path.join(PROJECT_ROOT, "hospital_docs")
                os.makedirs(save_dir, exist_ok=True)
                dest_path = os.path.join(save_dir, uploaded_h_file.name)
                try:
                    with open(dest_path, "wb") as f:
                        f.write(uploaded_h_file.getbuffer())
                    st.success(f"Uploaded '{uploaded_h_file.name}' to hospital_docs/.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save document: {e}")
                    
        with upcol2:
            st.write("👤 **Upload External Patient MD (MD)**")
            uploaded_p_file = st.file_uploader("Upload to patient_documents/", type=["md"])
            if uploaded_p_file is not None:
                p_id_input = st.text_input("Enter Patient ID for this document (e.g. P001)")
                if st.button("Save to Patient Record"):
                    if not p_id_input.strip():
                        st.error("Patient ID is required to save.")
                    else:
                        save_dir = os.path.join(PROJECT_ROOT, "patient_documents", p_id_input.strip())
                        os.makedirs(save_dir, exist_ok=True)
                        dest_path = os.path.join(save_dir, uploaded_p_file.name)
                        try:
                            with open(dest_path, "wb") as f:
                                f.write(uploaded_p_file.getbuffer())
                            st.success(f"Uploaded '{uploaded_p_file.name}' to patient_documents/{p_id_input.strip()}/.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save patient document: {e}")
                            
        st.write("---")
        
        # Rebuild Index Actions
        st.subheader("Rebuild Vector Store Indexes")
        st.write("Rebuilding will clear the existing FAISS indexes and re-index all Markdown files in `patient_documents/` and PDF/MD files in `hospital_docs/` using the chunk size and chunk overlap configured in settings.")
        if st.button("Rebuild Knowledge Base Now"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_progress(percent: float, text: str):
                progress_bar.progress(percent)
                status_text.write(text)
                
            with st.spinner("Rebuilding knowledge base indices..."):
                try:
                    h_chunks, p_chunks = rebuild_knowledge_base(update_progress)
                    st.success(f"Successfully rebuilt! Indexed {h_chunks} hospital chunks and {p_chunks} patient chunks.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

    # -----------------------------
    # 7. WARD & BED MANAGEMENT
    # -----------------------------
    elif menu == "🛏 Ward & Bed Management":
        st.title("🛏 Ward & Bed Occupancy")
        st.markdown("Displaying ward capacities, current occupied count, and patient assignments.")
        
        try:
            bed_info_json = bed_availability_lookup.invoke({})
            bed_info = json.loads(bed_info_json)
        except Exception as e:
            st.error(f"Failed to load bed availability backend: {e}")
            bed_info = {}

        patients = get_all_patients()
        
        for ward, info in bed_info.items():
            st.markdown(f"### 🚪 Ward: {ward}")
            
            # Calculate availability card
            tot_cap = info.get("total_capacity", "Unknown")
            occ_cnt = info.get("occupied_count", 0)
            av_cnt = info.get("available_count", "Unknown")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Beds Capacity", tot_cap)
            with c2:
                st.metric("Occupied Beds", occ_cnt)
            with c3:
                st.metric("Available Beds", av_cnt)
                
            # Patients assigned to this ward from SQLite
            ward_patients = [p for p in patients if p.get("ward") == ward]
            st.markdown("**Patient Allocations:**")
            if ward_patients:
                allocation_data = []
                for p in ward_patients:
                    allocation_data.append({
                        "Patient ID": p.get("patient_id"),
                        "Patient Name": p.get("name"),
                        "Bed Number": p.get("bed_number"),
                        "Current Diagnosis": p.get("diagnosis"),
                        "Status": p.get("current_status")
                    })
                st.dataframe(allocation_data, use_container_width=True)
            else:
                st.info("No patients currently assigned to this ward.")
            st.write("---")

    # -----------------------------
    # 8. SETTINGS
    # -----------------------------
    elif menu == "⚙ Settings":
        st.title("⚙ System Configurations")
        st.markdown("Configure AI assistant models, indexing parameters, and credentials locally.")
        
        with st.form("settings_form"):
            st.write("### Credentials")
            new_key = st.text_input("Groq API Key", value=api_key, type="password", help="Input your Groq credentials here. API key is saved locally and never shared.")
            
            st.write("### Model Configurations")
            new_model = st.selectbox(
                "AI Assistant Model Selection",
                ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
                index=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"].index(model_name)
            )
            new_temp = st.slider("Temperature (Creativity/Precision)", min_value=0.0, max_value=1.0, value=float(settings.get("temperature", 0.1)), step=0.05)
            
            st.write("### RAG & Indexing configurations")
            new_chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=int(settings.get("chunk_size", 1000)), step=100)
            new_chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=2000, value=int(settings.get("chunk_overlap", 200)), step=50)
            new_top_k = st.number_input("Top-K Document Retrieval Count", min_value=1, max_value=20, value=int(settings.get("top_k_retrieval", 3)))
            
            save_btn = st.form_submit_button("Save Configurations")
            
            if save_btn:
                new_settings = {
                    "groq_api_key": new_key.strip(),
                    "model_selection": new_model,
                    "temperature": new_temp,
                    "chunk_size": new_chunk_size,
                    "chunk_overlap": new_chunk_overlap,
                    "top_k_retrieval": new_top_k
                }
                try:
                    save_settings(new_settings)
                    st.success("Configurations successfully saved locally!")
                    st.info("Rebuilding the Knowledge Base is recommended if Chunk Size or Chunk Overlap was changed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save settings: {e}")

    # -----------------------------
    # 9. ABOUT VERAOPS
    # -----------------------------
    elif menu == "ℹ About VeraOps":
        st.title("ℹ About VeraOps")
        
        with st.container(border=True):
            st.markdown('<div class="card-header">Project Overview</div>', unsafe_allow_html=True)
            st.write("""
            **VeraOps** is an advanced, Agentic AI Hospital Intelligence and Decision Support Assistant. It is designed to streamline clinical workflows by integrating structured patient records management, multi-agent reasoning, and RAG pipelines. It helps clinical teams query medical guidelines, analyze historical profiles, and coordinate ward allocations in real time.
            """)
            
        st.write("")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown('<div class="card-header">Core Tech Stack</div>', unsafe_allow_html=True)
                st.write("""
                - **Language**: Python
                - **UI Engine**: Streamlit
                - **Agent framework**: LangGraph & LangChain
                - **Inference Engine**: Groq Cloud Services
                - **Database**: SQLite (SQL patient stores)
                - **Vector Indices**: FAISS L2 Vector Databases
                - **Sentence Model**: sentence-transformers/all-MiniLM-L6-v2
                - **Parsing Engine**: PyMuPDF
                """)
                
        with col2:
            with st.container(border=True):
                st.markdown('<div class="card-header">Architecture Overview</div>', unsafe_allow_html=True)
                st.write("""
                - **Hospital SOP RAG**: Indexes and retrieves operational guidelines and guidelines files.
                - **Patient Case RAG**: Indexes historical markdown documents for patient history search.
                - **LangGraph Supervisor Route**: Orchestrates the assistant loop to select tools, aggregate clinical results, and compile markdown structured outputs.
                """)
                
        st.write("")
        
        with st.container(border=True):
            st.markdown('<div class="card-header">Metadata & Developer Info</div>', unsafe_allow_html=True)
            st.write("- **Version**: 1.2.0")
            st.write("- **Developer**: Sanskriti / Hospital Ops Team")
            st.write("- **GitHub**: [github.com/hospital-ops-solutions/veraops-agent](https://github.com)")

        st.write("---")
        st.subheader("System Overview")
        
        # Pull live statistics
        from rag.document_store import get_docs_index
        from rag.patient_store import get_patients_index
        from rag.ingestion import MANIFEST_PATH
        
        try:
            kb_stats = get_knowledge_base_stats()
            h_v_count = get_docs_index().ntotal
            p_v_count = get_patients_index().ntotal
            v_dim = get_docs_index().d if h_v_count > 0 else 384
            status_v = "Rebuild Needed" if check_needs_rebuild() else "Up to Date"
            
            if os.path.exists(MANIFEST_PATH):
                last_up = datetime.fromtimestamp(os.path.getmtime(MANIFEST_PATH)).strftime('%Y-%m-%d %H:%M:%S')
            else:
                last_up = "N/A"
        except Exception:
            kb_stats = {"hospital_docs_count": 0, "patient_docs_count": 0}
            h_v_count, p_v_count = 0, 0
            v_dim = 384
            status_v = "Error"
            last_up = "N/A"
            
        patients_all = get_all_patients()
        tot_pts = len(patients_all)
        adm_pts = len([p for p in patients_all if p.get("current_status") != "Discharged"])
        dis_pts = len([p for p in patients_all if p.get("current_status") == "Discharged"])
        depts = sorted(list({p.get("department") for p in patients_all if p.get("department")}))
        
        try:
            bed_stats = get_bed_status_stats()
            occ_beds = bed_stats["occupied_beds"]
            avail_beds = bed_stats["available_beds"]
        except Exception:
            occ_beds = 0
            avail_beds = 1000
            
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            with st.container(border=True):
                st.markdown('<div class="card-header">💻 PROJECT CONFIGURATION</div>', unsafe_allow_html=True)
                st.write(f"- **VeraOps Version**: `1.2.0` (Production)")
                st.write(f"- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`")
                st.write(f"- **LLM Model**: `{model_name}`")
                st.write(f"- **Vector Database**: `FAISS` (Facebook AI Similarity Search)")
                st.write(f"- **Database Engine**: `SQLite 3`")
                st.write(f"- **AI Framework**: `LangGraph & LangChain` (Supervisor router + tools workflow)")
                
            with st.container(border=True):
                st.markdown('<div class="card-header">📚 KNOWLEDGE BASE METRICS</div>', unsafe_allow_html=True)
                st.write(f"- **Hospital Guidelines Documents**: `{kb_stats.get('hospital_docs_count', 0)}` files")
                st.write(f"- **Patient Case History Folders**: `{kb_stats.get('patient_docs_count', 0)}` folders")
                st.write(f"- **Total Document Assets**: `{kb_stats.get('hospital_docs_count', 0) + kb_stats.get('patient_docs_count', 0)}` files")
                st.write(f"- **Supported Formats**: `PDF, DOCX, TXT, MD` (Markdown/Text clinical formats)")
                
            with st.container(border=True):
                st.markdown('<div class="card-header">🧬 VECTOR STORE METRICS</div>', unsafe_allow_html=True)
                st.write(f"- **Hospital Guidelines Vector Count**: `{h_v_count}` vectors")
                st.write(f"- **Patient History Vector Count**: `{p_v_count}` vectors")
                st.write(f"- **Total Index Chunks**: `{h_v_count + p_v_count}` vectors")
                st.write(f"- **Embedding Dimensions**: `{v_dim}` floats")
                st.write(f"- **Vector Store Index Status**: `{status_v}`")
                st.write(f"- **Last Index Sync Update**: `{last_up}`")

        with col_s2:
            with st.container(border=True):
                st.markdown('<div class="card-header">🛢️ SQL DATABASE STATS</div>', unsafe_allow_html=True)
                st.write(f"- **Total Registered Patients**: `{tot_pts}` patients")
                st.write(f"- **Active Admitted Patients**: `{adm_pts}` patients")
                st.write(f"- **Discharged Patients**: `{dis_pts}` patients")
                st.write(f"- **Registered Hospital Wards**: `10` wards (100 beds/ward)")
                st.write(f"- **Total Wards Capacity**: `1000` hospital beds")
                st.write(f"- **Occupied Wards Beds**: `{occ_beds}` beds")
                st.write(f"- **Available Wards Beds**: `{avail_beds}` beds")
                st.write(f"- **Hospital Departments**: {', '.join([f'`{d}`' for d in depts]) if depts else 'None'}")
                
            with st.container(border=True):
                st.markdown('<div class="card-header">🛠️ CURRENT PROJECT DIRECTORIES</div>', unsafe_allow_html=True)
                st.write("- `hospital_docs/` (guidelines SOP storage)")
                st.write("- `patient_documents/` (individual patient timeline reports)")
                st.write("- `vector_store/` (FAISS indexes and serialization)")
                st.write("- `db/` (SQLite database helpers)")
                st.write("- `agents/` (LangGraph state and nodes definition)")
                st.write("- `tools/` (LangChain-compliant agent tools)")
                st.write("- `scripts/` (database seed tools)")
                st.write("- `data/` (raw JSON configuration data)")
                
        st.write("---")
        st.write("### 🏗️ Agentic RAG Pipeline Architecture")
        st.info("""
        ```
         [ User Query ]
               │
               ▼
        [ LangGraph Supervisor Agent ] ──(State Memory)
               │
          ┌────┴────────────────────────┐
          ▼                             ▼
        [ Patient Retriever ]       [ Hospital Retriever ]
        (FAISS Case Files)          (FAISS Guideline SOPs)
          │                             │
          └────┬────────────────────────┘
               ▼
        [ Context Aggregation ]
               │
               ▼
        [ Groq Inference LLM ]
               │
               ▼
        [ Grounded Clinical Response ] ──(How this answer was generated)
        ```
        """)