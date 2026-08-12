import os
import json
import io
import time
from datetime import datetime
import streamlit as st
from PIL import Image, ImageDraw
import cv2

# Import modular engines
import ai_engine as ai
from attendance_qr import AntiProxyAttendanceManager
from proctoring import ProctoringManager
from flashcard_engine import FlashcardDeck
from attentiveness_cv import OpenCVAttentivenessTracker, calculate_aggregated_performance_index
from analytics_recommendation import AnalyticsRecommendationEngine
from ephemeral_storage import EphemeralStorageManager

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Classroom AI Assistant - Enterprise Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM GLASSMORPHISM & HIGH CONTRAST CSS STYLING ---
st.markdown("""
<style>
    /* Dark Theme Core Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* HIGH CONTRAST INPUT & DROPDOWN READABILITY FIX */
    .stTextInput input, 
    .stTextArea textarea, 
    .stSelectbox select,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > input {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: 2px solid #6366f1 !important;
        font-size: 1rem !important;
    }

    /* Popover/Dropdown Menu Items Readability */
    div[data-baseweb="popover"] div,
    div[data-baseweb="menu"] li,
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        font-weight: 600 !important;
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #64748b !important;
        font-weight: 500 !important;
    }

    .stTextInput label, .stTextArea label, .stSelectbox label, .stFileUploader label, .stAudioInput label, .stNumberInput label {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        margin-bottom: 4px !important;
    }

    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .demo-credentials-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border: 2px dashed #818cf8;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 20px;
        text-align: center;
    }

    .broadcast-card {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border-left: 6px solid #ec4899;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }

    .timetable-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .gradient-header {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }

    .sub-gradient-header {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.35rem;
    }

    .badge-live {
        background-color: rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    .badge-approved {
        background-color: rgba(16, 185, 129, 0.25);
        color: #6ee7b7;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .badge-rejected {
        background-color: rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .timestamp-chip {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 8px;
        font-family: monospace;
        font-weight: bold;
        font-size: 0.9rem;
    }

    .doubt-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #a855f7;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .reply-card {
        background: rgba(16, 185, 129, 0.15);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# --- INITIALIZE SESSION STATE ---
def init_session_state():
    if "users" not in st.session_state:
        st.session_state["users"] = {
            "teacher": {"password": "123", "role": "Teacher", "name": "Prof. Alan Turing"},
            "student": {"password": "123", "role": "Student", "name": "Alex Student"}
        }

    if "authenticated_user" not in st.session_state:
        st.session_state["authenticated_user"] = None
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = None
    if "user_name" not in st.session_state:
        st.session_state["user_name"] = None

    if "timetable" not in st.session_state:
        st.session_state["timetable"] = [
            {"id": 1, "subject": "Physics 101: Classical Mechanics", "start": "09:00 AM", "end": "10:00 AM", "duration": "60 mins", "status": "🔴 LIVE NOW"},
            {"id": 2, "subject": "Mathematics: Calculus & Derivatives", "start": "10:30 AM", "end": "11:30 AM", "duration": "60 mins", "status": "⏳ UPCOMING"},
            {"id": 3, "subject": "Computer Science: Data Structures", "start": "01:00 PM", "end": "02:15 PM", "duration": "75 mins", "status": "⏳ UPCOMING"}
        ]

    if "media_gallery" not in st.session_state:
        st.session_state["media_gallery"] = []
    if "lecture_data" not in st.session_state:
        st.session_state["lecture_data"] = None
    if "board_data" not in st.session_state:
        st.session_state["board_data"] = None

    if "doubts" not in st.session_state:
        st.session_state["doubts"] = []
    if "published_quizzes" not in st.session_state:
        st.session_state["published_quizzes"] = []
    if "broadcasts" not in st.session_state:
        st.session_state["broadcasts"] = []
    if "tutor_chat" not in st.session_state:
        st.session_state["tutor_chat"] = []
    if "student_lang" not in st.session_state:
        st.session_state["student_lang"] = "Hindi"
    if "student_translated_cache" not in st.session_state:
        st.session_state["student_translated_cache"] = {}

    # New Module Managers
    if "attendance_manager" not in st.session_state:
        st.session_state["attendance_manager"] = AntiProxyAttendanceManager()
    if "attendance_records" not in st.session_state:
        st.session_state["attendance_records"] = []

    if "proctoring_logs" not in st.session_state:
        st.session_state["proctoring_logs"] = []

    if "flashcard_deck" not in st.session_state:
        st.session_state["flashcard_deck"] = FlashcardDeck()

    if "attentiveness_tracker" not in st.session_state:
        st.session_state["attentiveness_tracker"] = OpenCVAttentivenessTracker()
    if "attentiveness_score" not in st.session_state:
        st.session_state["attentiveness_score"] = 92.0

    if "quiz_score_pct" not in st.session_state:
        st.session_state["quiz_score_pct"] = 100.0


init_session_state()

# Auto-purge expired session data > 24 hours
EphemeralStorageManager.auto_purge_expired_session_data(st.session_state)


# --- HELPER TO GET GROQ CLIENT ---
def get_client(sidebar_api_key=""):
    key = sidebar_api_key.strip() or os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        st.error("🔑 Groq API Key missing! Please set GROQ_API_KEY in your .env file or enter it in the sidebar.")
        st.stop()
    return ai.get_groq_client(key)


# --- SAMPLE WHITEBOARD GENERATOR ---
def create_sample_whiteboard():
    img = Image.new("RGB", (900, 500), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)
    for y in range(0, 500, 40):
        draw.line([(0, y), (900, y)], fill=(225, 232, 240), width=1)
    for x in range(0, 900, 40):
        draw.line([(x, 0), (x, 500)], fill=(225, 232, 240), width=1)

    draw.text((40, 40), "Class: Physics 101 - Newton's Laws & Energy", fill=(15, 23, 42))
    draw.text((40, 100), "1. Force Equation: F = m * a", fill=(30, 58, 138))
    draw.text((40, 160), "2. Kinetic Energy: K = (1/2) * m * v^2", fill=(124, 45, 18))
    draw.text((40, 220), "3. Work Done Integral: W = integral(F dx)", fill=(15, 118, 110))
    draw.text((40, 290), "Einstein Mass-Energy Equivalence: E = m * c^2", fill=(157, 23, 77))
    draw.text((40, 360), "Key Note: Mass is in kg, acceleration is in m/s^2.", fill=(30, 41, 59))
    return img


# ==========================================
# AUTHENTICATION PAGE
# ==========================================
def render_auth_page():
    st.markdown("<h1 class='gradient-header' style='text-align: center;'>🎓 Smart Classroom AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #cbd5e1; font-size: 1.15rem; font-weight: 500;'>Real-time AI Audio Translation • Dynamic QR Geofencing • OpenCV Attentiveness • Spaced Repetition Flashcards</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2.2, 1])
    with col2:
        st.markdown("""
        <div class='demo-credentials-box'>
            <h4 style='margin:0; color:#cbd5e1;'>⚡ Predefined Hackathon Demo Credentials</h4>
            <p style='margin:6px 0 0 0; color:#f8fafc; font-size:1.05rem;'>
                <b>👨‍🏫 Teacher Role:</b> Username: <code style='color:#a5b4fc; background:rgba(0,0,0,0.4); padding:2px 6px; border-radius:4px;'>teacher</code> | Password: <code style='color:#a5b4fc; background:rgba(0,0,0,0.4); padding:2px 6px; border-radius:4px;'>123</code><br>
                <b>🎓 Student Role:</b> Username: <code style='color:#6ee7b7; background:rgba(0,0,0,0.4); padding:2px 6px; border-radius:4px;'>student</code> | Password: <code style='color:#6ee7b7; background:rgba(0,0,0,0.4); padding:2px 6px; border-radius:4px;'>123</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(["🔒 Account Login", "✨ Register New Account"])

        with tab_login:
            st.subheader("Login to Smart Classroom")
            login_username = st.text_input("Username", key="login_user", placeholder="teacher or student")
            login_password = st.text_input("Password", type="password", key="login_pass", placeholder="123")

            if st.button("🚀 Sign In", use_container_width=True, type="primary"):
                user_db = st.session_state["users"]
                if login_username in user_db and user_db[login_username]["password"] == login_password:
                    st.session_state["authenticated_user"] = login_username
                    st.session_state["user_role"] = user_db[login_username]["role"]
                    st.session_state["user_name"] = user_db[login_username]["name"]
                    st.success(f"Logged in as {st.session_state['user_name']} ({st.session_state['user_role']})!")
                    st.rerun()
                else:
                    st.error("Invalid username or password!")

            st.markdown("---")
            st.markdown("<p style='text-align:center; font-weight:bold; color:#a5b4fc;'>⚡ Quick 1-Click Demo Login:</p>", unsafe_allow_html=True)
            pcol1, pcol2 = st.columns(2)
            if pcol1.button("👨‍🏫 Demo as Teacher", use_container_width=True):
                st.session_state["authenticated_user"] = "teacher"
                st.session_state["user_role"] = "Teacher"
                st.session_state["user_name"] = "Prof. Alan Turing"
                st.rerun()
            if pcol2.button("🎓 Demo as Student", use_container_width=True):
                st.session_state["authenticated_user"] = "student"
                st.session_state["user_role"] = "Student"
                st.session_state["user_name"] = "Alex Student"
                st.rerun()

        with tab_signup:
            st.subheader("Create a New Account")
            new_name = st.text_input("Full Name", key="signup_name")
            new_username = st.text_input("Choose Username", key="signup_user")
            new_password = st.text_input("Choose Password", type="password", key="signup_pass")
            new_role = st.selectbox("Select Role", ["Student", "Teacher"], key="signup_role")

            if st.button("✨ Create Account", use_container_width=True):
                if not new_username or not new_password or not new_name:
                    st.warning("Please fill in all fields.")
                elif new_username in st.session_state["users"]:
                    st.error("Username already exists!")
                else:
                    st.session_state["users"][new_username] = {
                        "password": new_password,
                        "role": new_role,
                        "name": new_name
                    }
                    st.session_state["authenticated_user"] = new_username
                    st.session_state["user_role"] = new_role
                    st.session_state["user_name"] = new_name
                    st.success("Account created successfully!")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# MAIN APPLICATION CONTROLLER
# ==========================================
def main():
    if not st.session_state["authenticated_user"]:
        render_auth_page()
        return

    # Render Proctoring Tracker JS for Student session
    if st.session_state["user_role"] == "Student":
        ProctoringManager.render_client_proctor_tracker(st.session_state["authenticated_user"])

    with st.sidebar:
        st.markdown("### 🎓 Smart Classroom Pro")
        st.markdown(f"🟢 **LIVE CLASSROOM ACTIVE**")
        st.markdown(f"**User:** {st.session_state['user_name']}")
        role_tag = "👨‍🏫 Teacher Command Center" if st.session_state["user_role"] == "Teacher" else "🎓 Student Learning Hub"
        st.markdown(f"**Role:** `{role_tag}`")

        st.markdown("---")
        sidebar_key = st.text_input("Groq API Key (Optional Override)", type="password", help="Leave blank if set in .env")

        if st.session_state["user_role"] == "Student":
            st.markdown("---")
            st.markdown("#### 🌐 Student Target Language")
            st.session_state["student_lang"] = st.selectbox(
                "Select Native Language for Dual Text & Voice",
                ["Hindi", "Bangla", "Arabic", "Spanish", "French", "German", "English"],
                index=["Hindi", "Bangla", "Arabic", "Spanish", "French", "German", "English"].index(st.session_state.get("student_lang", "Hindi"))
            )

        st.markdown("---")
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state["authenticated_user"] = None
            st.session_state["user_role"] = None
            st.session_state["user_name"] = None
            st.rerun()

    if st.session_state["user_role"] == "Teacher":
        render_teacher_dashboard(sidebar_key)
    else:
        render_student_dashboard(sidebar_key)


# ==========================================
# TEACHER DASHBOARD
# ==========================================
def render_teacher_dashboard(sidebar_key):
    st.markdown("<h2 class='gradient-header'>👨‍🏫 Teacher Command Center</h2>", unsafe_allow_html=True)
    st.markdown("Real-time Audio Translation, Dynamic QR Geofencing, Proctored Tab Tracker, Heatmap Analytics & Quiz Publisher.")
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎙️ 1. Audio Translation & Board Vision",
        "🛡️ 2. Anti-Proxy Dynamic QR & Geofence",
        "👁️ 3. Proctored Alert Feed",
        "📊 4. Engagement Heatmap & Quiz Creator",
        "💬 5. Multilingual Doubt Inbox",
        "🧹 6. Ephemeral Purge Controller"
    ])

    client = get_client(sidebar_key)

    # --- TAB 1: AUDIO TRANSLATION & BOARD SCANNER ---
    with tab1:
        col_lec, col_board = st.columns(2)

        with col_lec:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='sub-gradient-header'>🎙️ Live Mic Audio Recording & Transcription</h3>", unsafe_allow_html=True)

            recorded_audio = st.audio_input("Record Live Teacher Mic Audio:")
            uploaded_audio = st.file_uploader("Upload Audio/Video File (.mp3, .wav, .mp4)", type=["mp3", "wav", "m4a", "mp4", "webm"])
            transcript_caption = st.text_input("Teacher Caption / Note for Audio:", placeholder="E.g., Physics 101 Lecture on Newton's Laws", key="transcript_caption_input")

            transcript_input = st.text_area(
                "Or Paste / Edit Lecture Transcript (English):",
                height=130,
                placeholder="Welcome class! Today we will learn about Newton's Second Law of Motion: F = m * a...",
                key="teacher_transcript_input"
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("⚡ Process & Publish (English)", use_container_width=True, type="primary"):
                    text_to_process = transcript_input.strip()

                    audio_source = recorded_audio or uploaded_audio
                    if audio_source:
                        with st.spinner("Transcribing live audio via Groq Whisper API..."):
                            transcribed_text = ai.transcribe_audio_file(client, audio_source)
                            if transcribed_text:
                                text_to_process = transcribed_text
                                st.info(f"Transcribed Audio: {transcribed_text[:120]}...")

                    if not text_to_process:
                        st.warning("Please record mic audio, upload a file, or type a transcript.")
                    else:
                        with st.spinner("Groq AI generating structured notes & chapters..."):
                            try:
                                data = ai.process_lecture_transcript(client, text_to_process, target_language="English")
                                st.session_state["lecture_data"] = data
                                st.success("Lecture published in English!")
                            except Exception as e:
                                st.error(f"Error processing lecture: {e}")

            with col_btn2:
                if st.button("📋 Load Physics Demo Preset", use_container_width=True):
                    demo_text = """Welcome to Physics 101. Today we are exploring Classical Mechanics and Newton's Laws. 
First, let us examine Newton's Second Law of Motion: Force equals mass multiplied by acceleration, or F = m * a.
Next, at timestamp 04:15, we define Kinetic Energy as K = 0.5 * m * v^2, representing work needed to accelerate a body of a given mass.
Finally, at timestamp 09:30, we touch upon Einstein's Mass-Energy Equivalence formula: E = m * c^2, showing that energy and mass are interchangeable."""
                    with st.spinner("Loading demo physics lecture..."):
                        data = ai.process_lecture_transcript(client, demo_text, target_language="English")
                        st.session_state["lecture_data"] = data
                        st.success("Sample Physics Lecture Loaded!")
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        with col_board:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='sub-gradient-header'>📷 Smart Board Multimodal Vision Scanner</h3>", unsafe_allow_html=True)

            uploaded_img = st.file_uploader("Upload Whiteboard / Notebook Image", type=["jpg", "jpeg", "png"])
            board_caption = st.text_input("Teacher Caption for Board Image:", placeholder="E.g., Whiteboard diagram for Newton's 2nd Law", key="board_caption_input")

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("🔍 Scan & Publish Board", use_container_width=True, type="primary"):
                    if uploaded_img is None:
                        st.warning("Please upload an image first.")
                    else:
                        image_pil = Image.open(uploaded_img)
                        with st.spinner("Groq Vision AI scanning board, extracting LaTeX & flowcharts..."):
                            try:
                                board_res = ai.process_smartboard_image(client, image_pil, target_language="English")
                                st.session_state["board_data"] = board_res
                                st.success("Board photo, LaTeX & Flowchart analysis published!")
                            except Exception as e:
                                st.error(f"Vision OCR Error: {e}")

            with col_b2:
                if st.button("🖼️ Load Sample Board Preset", use_container_width=True):
                    sample_img = create_sample_whiteboard()
                    with st.spinner("Groq Vision AI processing sample whiteboard..."):
                        try:
                            board_res = ai.process_smartboard_image(client, sample_img, target_language="English")
                            st.session_state["board_data"] = board_res
                            st.success("Sample Board & LaTeX Published!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Vision OCR Error: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        # Active Outputs Display
        if st.session_state["lecture_data"] or st.session_state["board_data"]:
            st.markdown("---")
            st.markdown("<h3 class='sub-gradient-header'>📊 Active Class Content & LaTeX Formulas</h3>", unsafe_allow_html=True)

            out_col1, out_col2 = st.columns(2)
            with out_col1:
                if st.session_state["lecture_data"]:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown("#### 🎯 Structured Class Notes (Markdown & LaTeX)")
                    st.write(st.session_state["lecture_data"].get("structured_notes", st.session_state["lecture_data"].get("summary", "")))

                    st.markdown("#### ⏱️ Smart Chapters Timeline")
                    for ch in st.session_state["lecture_data"].get("chapters", []):
                        st.markdown(f"<span class='timestamp-chip'>{ch.get('timestamp','00:00')}</span> **{ch.get('title','')}** - {ch.get('description','')}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            with out_col2:
                if st.session_state["board_data"]:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown("#### 🧮 Smart Board LaTeX Formulas & Flowchart Analysis")
                    st.write("**Extracted Text:**", st.session_state["board_data"].get("extracted_text", ""))
                    for f in st.session_state["board_data"].get("latex_formulas", []):
                        st.latex(f)
                    st.markdown("**Diagram/Flowchart Analysis:**")
                    st.write(st.session_state["board_data"].get("diagram_analysis", st.session_state["board_data"].get("explanation", "")))
                    st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 2: DYNAMIC QR & GPS GEOFENCE ATTENDANCE ---
    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-gradient-header'>🛡️ Next-Gen Dynamic TOTP QR & GPS Geofence Engine</h3>", unsafe_allow_html=True)
        st.write("Dynamic QR Code automatically refreshes every 10–15 seconds with a 25-second server tolerance window. Student GPS coordinates are validated against classroom boundaries.")

        col_qr, col_geo = st.columns([1.2, 1])

        att_mgr = st.session_state["attendance_manager"]

        with col_qr:
            st.markdown("#### 📱 Live Dynamic TOTP QR Code")
            qr_base64 = att_mgr.generate_qr_image_base64()
            payload_info = att_mgr.get_current_qr_payload()

            st.image(f"data:image/png;base64,{qr_base64}", width=240)
            st.markdown(f"**Current 6-Digit OTP Code:** `<h3 style='color:#818cf8; display:inline;'>{payload_info['otp_code']}</h3>`", unsafe_allow_html=True)
            st.caption(f"⏳ Dynamic Refresh in: {payload_info['time_remaining']} seconds (25s Tolerance Window Active)")

            if st.button("🔄 Force Refresh Dynamic Seed", type="secondary"):
                st.session_state["attendance_manager"] = AntiProxyAttendanceManager()
                st.rerun()

        with col_geo:
            st.markdown("#### 📍 Classroom GPS Geofence Configurator")
            c_lat = st.number_input("Classroom Latitude:", value=att_mgr.classroom_lat, format="%.4f")
            c_lon = st.number_input("Classroom Longitude:", value=att_mgr.classroom_lon, format="%.4f")
            c_rad = st.number_input("Allowed Geofence Radius (Meters):", value=att_mgr.max_radius, step=5.0)

            if st.button("💾 Save Geofence Settings", type="primary"):
                att_mgr.set_classroom_location(c_lat, c_lon, c_rad)
                st.success("Classroom Geofence updated!")

        st.markdown("---")
        st.markdown("#### 📋 Real-Time Attendance Log")
        records = st.session_state["attendance_records"]
        if not records:
            st.info("No student attendance scans recorded yet today.")
        else:
            for r in reversed(records):
                badge_class = "badge-approved" if r["status"] == "APPROVED" else "badge-rejected"
                st.markdown(f"""
                <div class='timetable-card'>
                    <div style='display:flex; justify-content:space-between;'>
                        <b>Student: {r['student_name']} ({r['timestamp']})</b>
                        <span class='{badge_class}'>{r['status']}</span>
                    </div>
                    <p style='margin:4px 0 0 0; color:#cbd5e1; font-size:0.9rem;'>
                        📍 Distance: <b>{r['distance_meters']}m</b> | OTP Valid: {r['is_otp_valid']} | Geofence Valid: {r['is_geofence_valid']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 3: PROCTORED ALERT FEED ---
    with tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-gradient-header'>👁️ Proctored Focus & Tab Switch Incident Feed</h3>", unsafe_allow_html=True)
        st.write("Live alerts triggered when students switch tabs, lose window focus, or resize browser below allowed aspect ratio.")

        logs = st.session_state["proctoring_logs"]

        col_sim1, col_sim2 = st.columns([1, 3])
        with col_sim1:
            if st.button("🧪 Simulate Student Tab Switch Alert"):
                ProctoringManager.log_incident(
                    st.session_state["proctoring_logs"],
                    student_id="student_alex",
                    incident_type="TAB_SWITCH",
                    details="Student switched away from classroom tab to secondary application."
                )
                st.rerun()

            if st.button("🧪 Simulate Split-Screen Alert"):
                ProctoringManager.log_incident(
                    st.session_state["proctoring_logs"],
                    student_id="student_alex",
                    incident_type="SPLIT_SCREEN_DETECTION",
                    details="Window width dropped to 520px (Aspect Ratio: 0.95)."
                )
                st.rerun()

        with col_sim2:
            if not logs:
                st.info("No proctoring focus alerts recorded for current session.")
            else:
                for l in logs:
                    icon = "⚠️" if "TAB" in l["type"] else "📐"
                    st.markdown(f"""
                    <div style='background:rgba(239, 68, 68, 0.15); border-left:4px solid #ef4444; border-radius:8px; padding:12px; margin-bottom:8px;'>
                        <b>{icon} [{l['timestamp']}] Student: {l['student_id']} - {l['type']}</b><br>
                        <span style='color:#cbd5e1;'>{l['details']}</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 4: ENGAGEMENT HEATMAP & QUIZ PUBLISHER ---
    with tab4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-gradient-header'>📊 Teacher Engagement Heatmap & Quiz Publisher</h3>", unsafe_allow_html=True)

        # Plotly Engagement Heatmap
        fig = AnalyticsRecommendationEngine.create_teacher_engagement_heatmap([])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📝 Automated AI Quiz Generator")
        lecture_context = ""
        if st.session_state["lecture_data"]:
            lecture_context += st.session_state["lecture_data"].get("summary", "") + "\n" + st.session_state["lecture_data"].get("structured_notes", "")
        if st.session_state["board_data"]:
            lecture_context += "\n" + st.session_state["board_data"].get("extracted_text", "")

        if not lecture_context.strip():
            st.warning("Please process a lecture or scan a board first in Tab 1.")
        else:
            if st.button("⚡ Generate AI Quiz from Lecture Context", type="primary"):
                with st.spinner("Groq AI generating MCQs..."):
                    quiz = ai.generate_quiz(client, lecture_context, num_questions=3)
                    st.session_state["draft_quiz"] = quiz
                    st.success("Generated 3 Quiz Questions!")

            if "draft_quiz" in st.session_state and st.session_state["draft_quiz"]:
                st.markdown("#### 📋 Draft Questions Preview")
                for q in st.session_state["draft_quiz"]:
                    st.markdown(f"**Q{q['id']}: {q['question']}**")
                    for opt in q["options"]:
                        st.write(f"- {opt}")
                    st.info(f"**Correct Answer:** {q['answer']} | **Explanation:** {q['explanation']}")
                    st.markdown("---")

                if st.button("🚀 Publish Quiz to Students", type="primary"):
                    st.session_state["published_quizzes"] = st.session_state["draft_quiz"]
                    st.success("Quiz published to student dashboard!")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 5: MULTILINGUAL DOUBT INBOX ---
    with tab5:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-gradient-header'>💬 Anonymous Student Doubts Inbox</h3>", unsafe_allow_html=True)

        doubts = st.session_state["doubts"]
        if not doubts:
            st.info("No student doubts submitted yet.")
        else:
            for idx, d in enumerate(doubts):
                st.markdown("<div class='doubt-card'>", unsafe_allow_html=True)
                st.markdown(f"**From:** {d['student_alias']} | **Native Language:** {d['native_lang']}")
                st.markdown(f"**Original Doubt:** *\"{d['doubt_raw']}\"*")
                st.markdown(f"**🇬🇧 Translated English:** **{d['doubt_en']}**")

                if d.get("reply_en"):
                    st.markdown(f"<div class='reply-card'>**Your Reply:** {d['reply_en']}<br>**Translated ({d['native_lang']}):** {d['reply_translated']}</div>", unsafe_allow_html=True)
                else:
                    reply_input = st.text_input("Write reply in English:", key=f"t_reply_{idx}")
                    if st.button(f"📤 Send Answer to {d['student_alias']}", key=f"btn_r_{idx}"):
                        if reply_input.strip():
                            with st.spinner(f"Translating into {d['native_lang']} & synthesizing voice..."):
                                res = ai.translate_teacher_reply_to_student(client, reply_input, d['native_lang'])
                                doubts[idx]["reply_en"] = reply_input
                                doubts[idx]["reply_translated"] = res["translated_text"]
                                doubts[idx]["reply_audio"] = res["audio_bytes"]
                                st.session_state["doubts"] = doubts
                                st.success("Answer sent with localized voice audio!")
                                st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 6: EPHEMERAL PURGE CONTROLLER ---
    with tab6:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-gradient-header'>🧹 24-Hour Ephemeral Storage & Purge Manager</h3>", unsafe_allow_html=True)
        st.write("Automatically purges raw media files and high-frequency audio logs older than 24 hours to optimize database costs while preserving summary notes and analytics metrics.")

        last_stats = st.session_state.get("last_auto_purge_stats", {})
        st.write(f"**Last Auto-Purge Run:** {last_stats.get('timestamp', 'Just Now')}")
        st.write(f"**Items Purged:** {last_stats.get('purged_media_count', 0)} raw media objects")

        if st.button("⚡ Force Purge Media > 24 Hours Now", type="primary"):
            stats = EphemeralStorageManager.auto_purge_expired_session_data(st.session_state)
            st.success(f"Purged {stats['purged_media_count']} expired items to optimize storage!")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# STUDENT DASHBOARD
# ==========================================
def render_student_dashboard(sidebar_key):
    st.markdown("<h2 class='gradient-header'>🎓 Student Learning Hub</h2>", unsafe_allow_html=True)
    lang = st.session_state.get("student_lang", "Hindi")
    st.markdown(f"Dual Text & Voice Translation in **{lang}** • Anti-Proxy QR Scanner • OpenCV Attentiveness • Spaced Repetition Flashcards")
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        f"🌐 1. Dual Notes & Voice ({lang})",
        "📱 2. Anti-Proxy Attendance Scanner",
        "🃏 3. Spaced Repetition Flashcards",
        "👁️ 4. OpenCV Camera Attentiveness",
        "🎯 5. AI Study Recommendations",
        "✍️ 6. Quiz & Grounded AI Tutor"
    ])

    client = get_client(sidebar_key)

    # --- TAB 1: DUAL NOTES & VOICE TRANSLATION ---
    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='sub-gradient-header'>📄 Structured Class Notes & Localized Neural Voice ({lang})</h3>", unsafe_allow_html=True)

        lecture = st.session_state["lecture_data"]

        if not lecture:
            st.info("⏳ Waiting for teacher to process and publish today's lecture.")
        else:
            raw_text = lecture.get("structured_notes", "") or lecture.get("raw_transcript", "")

            cache_key = f"translated_{lang}"
            if cache_key not in st.session_state["student_translated_cache"]:
                with st.spinner(f"Translating full class notes into {lang} with LaTeX lock..."):
                    translated_paragraph = ai.translate_text(client, raw_text, lang)
                    st.session_state["student_translated_cache"][cache_key] = translated_paragraph

            translated_paragraph = st.session_state["student_translated_cache"].get(cache_key, raw_text)

            col_text, col_voice = st.columns([1.5, 1])

            with col_text:
                st.markdown(f"#### 📝 Complete Notes ({lang})")
                st.write(translated_paragraph)

            with col_voice:
                st.markdown("#### 🔊 Localized Neural Voice Player (`edge-tts`)")
                st.write(f"Listen to class notes in **{lang}**:")

                audio_cache_key = f"audio_{lang}"
                if audio_cache_key not in st.session_state["student_translated_cache"]:
                    if st.button(f"⚡ Generate Voice Audio in {lang}", type="primary"):
                        with st.spinner(f"Synthesizing Neural Speech in {lang}..."):
                            audio_bytes = ai.generate_audio(translated_paragraph, lang)
                            st.session_state["student_translated_cache"][audio_cache_key] = audio_bytes
                            st.rerun()

                if audio_cache_key in st.session_state["student_translated_cache"]:
                    st.audio(st.session_state["student_translated_cache"][audio_cache_key], format="audio/mp3")

            st.markdown("---")
            if lecture.get("chapters"):
                st.markdown("#### ⏱️ Smart Chapters Timeline")
                for ch in lecture.get("chapters", []):
                    st.markdown(f"<span class='timestamp-chip'>{ch.get('timestamp','00:00')}</span> **{ch.get('title','')}** - {ch.get('description','')}", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 2: ANTI-PROXY ATTENDANCE SCANNER ---
    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-gradient-header'>📱 Anti-Proxy Attendance Scanner (Dynamic QR + GPS Geofence)</h3>", unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)
        att_mgr = st.session_state["attendance_manager"]

        with col_s1:
            st.markdown("#### 🔑 Scan / Input Dynamic QR Code")
            scanned_otp = st.text_input("Enter 6-Digit Dynamic OTP Code from Board:", max_chars=6, placeholder="E.g. 482910")

            st.markdown("#### 📍 Student Device GPS Coordinates")
            s_lat = st.number_input("Your Device Latitude:", value=28.6139, format="%.4f")
            s_lon = st.number_input("Your Device Longitude:", value=77.2090, format="%.4f")

            if st.button("🚀 Verify & Submit Attendance", type="primary", use_container_width=True):
                if not scanned_otp or len(scanned_otp) < 6:
                    st.warning("Please enter a valid 6-digit dynamic OTP code.")
                else:
                    res = att_mgr.verify_attendance_scan(scanned_otp, s_lat, s_lon)
                    res["student_name"] = st.session_state["user_name"]

                    st.session_state["attendance_records"].append(res)

                    if res["status"] == "APPROVED":
                        st.success(f"✅ Attendance Approved! Location within {res['distance_meters']}m of classroom.")
                    else:
                        st.error(f"❌ Attendance Rejected: {', '.join(res['reasons'])}")

        with col_s2:
            st.markdown("#### 🛡️ Anti-Proxy Rules Active")
            st.info("""
            1. **Dynamic TOTP:** Verification code expires every 15s (25s tolerance window).
            2. **GPS Geofence Guard:** Scanner must be within 50 meters of the science lecture hall.
            """)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 3: SPACED REPETITION FLASHCARDS ---
    with tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-gradient-header'>🃏 Automated Flashcard Creator (SM-2 Spaced Repetition)</h3>", unsafe_allow_html=True)

        deck = st.session_state["flashcard_deck"]

        col_fc1, col_fc2 = st.columns([1, 2])

        with col_fc1:
            st.markdown("#### ➕ Create New Flashcards")
            if st.button("⚡ Generate Flashcards from Active Class Notes", type="primary", use_container_width=True):
                notes_text = ""
                if st.session_state["lecture_data"]:
                    notes_text = st.session_state["lecture_data"].get("structured_notes", "")
                if not notes_text and st.session_state["board_data"]:
                    notes_text = st.session_state["board_data"].get("extracted_text", "")

                if not notes_text:
                    st.warning("No processed notes found. Please process a lecture first!")
                else:
                    with st.spinner("AI generating Q&A flashcards..."):
                        cards = deck.create_flashcards_from_notes(client, notes_text)
                        st.success(f"Created {len(cards)} new flashcards!")
                        st.rerun()

            st.metric("Mastery Index", f"{deck.get_mastery_percentage()}%")
            st.write(f"Total Cards in Deck: {len(deck.cards)}")

        with col_fc2:
            st.markdown("#### 🎴 Review Due Cards")
            due_cards = deck.get_due_cards() or deck.cards

            if not due_cards:
                st.info("🎉 All caught up! No cards due for review right now.")
            else:
                card = due_cards[0]
                st.markdown(f"**Card ID #{card.id}** (Review Interval: {card.interval_days} days)")

                # Flip card simulator
                show_ans = st.checkbox("🔄 Show Answer", key=f"show_ans_{card.id}")

                st.markdown(f"""
                <div style='background:rgba(15,23,42,0.8); border:2px solid #818cf8; border-radius:12px; padding:24px; min-height:140px; text-align:center;'>
                    <h3 style='color:#f8fafc;'>{card.question}</h3>
                    {f"<h4 style='color:#6ee7b7; margin-top:16px;'>{card.answer}</h4>" if show_ans else "<p style='color:#94a3b8;'>(Check box to reveal answer)</p>"}
                </div>
                """, unsafe_allow_html=True)

                if show_ans:
                    st.write("**Rate Recall Ease (SM-2 Algorithm):**")
                    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
                    if r_col1.button("🔴 Again (0)", key="sm_0"):
                        card.update_sm2(0)
                        st.rerun()
                    if r_col2.button("🟠 Hard (2)", key="sm_2"):
                        card.update_sm2(2)
                        st.rerun()
                    if r_col3.button("🟢 Good (4)", key="sm_4"):
                        card.update_sm2(4)
                        st.rerun()
                    if r_col4.button("⚡ Easy (5)", key="sm_5"):
                        card.update_sm2(5)
                        st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 4: OPENCV CAMERA ATTENTIVENESS ---
    with tab4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-gradient-header'>👁️ OpenCV & MediaPipe Camera Attentiveness & Head Pose</h3>", unsafe_allow_html=True)
        st.write("Tracks head orientation (Yaw, Pitch, Roll) and eye gaze. If camera engagement is maintained over 15-second evaluation windows, frame is marked Attentive.")

        col_cv1, col_cv2 = st.columns(2)

        with col_cv1:
            st.markdown("#### 📹 Camera Stream / Fallback Simulation")
            cam_mode = st.radio("Camera Source:", ["Synthetic Demo Fallback", "Live Webcam Feed"], horizontal=True)

            tracker = st.session_state["attentiveness_tracker"]

            if cam_mode == "Synthetic Demo Fallback":
                is_att = st.checkbox("Simulate Attentive Head Pose", value=True)
                frame_bgr = tracker.generate_synthetic_head_frame(is_attentive=is_att)
                st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

                res = tracker.estimate_head_pose_and_gaze(frame_bgr)
                st.write(f"**Head Pose Yaw:** {res['yaw']}° | **Pitch:** {res['pitch']}° | **Attentive:** `{res['is_attentive']}`")
            else:
                img_input = st.camera_input("Capture Webcam Frame for Attentiveness Check")
                if img_input:
                    bytes_data = img_input.getvalue()
                    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    res = tracker.estimate_head_pose_and_gaze(cv_img)
                    st.write(f"**Status:** {res['reason']}")
                    st.write(f"**Yaw:** {res['yaw']}° | **Pitch:** {res['pitch']}° | **Roll:** {res['roll']}°")

        with col_cv2:
            st.markdown("#### 📈 Aggregated Performance Index Score")
            att_score = st.session_state["attentiveness_score"]
            q_score = st.session_state["quiz_score_pct"]
            att_rate = 100.0  # Attendance rate

            perf = calculate_aggregated_performance_index(att_score, q_score, att_rate)

            st.metric("Aggregated Performance Index", f"{perf['final_index']}%", delta=perf['badge'])
            st.write(f"- Attentiveness (40%): **{perf['breakdown']['attentiveness_weighted']} pts**")
            st.write(f"- Quiz Performance (35%): **{perf['breakdown']['quiz_weighted']} pts**")
            st.write(f"- Attendance Rate (25%): **{perf['breakdown']['attendance_weighted']} pts**")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 5: AI STUDY RECOMMENDATIONS ---
    with tab5:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-gradient-header'>🎯 AI Improvement Engine & Personalized Study Plan</h3>", unsafe_allow_html=True)
        st.write(f"Generates tailored recommendations based on quiz performance and attentiveness data in **{lang}**.")

        if st.button("⚡ Generate AI Improvement Plan", type="primary"):
            with st.spinner("Analyzing performance & constructing study plan..."):
                plan = AnalyticsRecommendationEngine.generate_personalized_study_plan(
                    client,
                    quiz_errors=["Q1: Newton 2nd Law equation"],
                    low_attention_timestamps=["04:15"],
                    student_doubts=[d.get("doubt_en", "") for d in st.session_state["doubts"]],
                    student_lang=lang
                )
                st.session_state["study_plan"] = plan

        if "study_plan" in st.session_state:
            plan = st.session_state["study_plan"]
            st.markdown(f"#### 🔍 Diagnostic: {plan.get('overall_diagnostic','')}")

            st.markdown("#### 📌 Recommended Topics to Review")
            for t in plan.get("recommended_topics", []):
                st.write(f"- **{t['topic']}** (`{t['priority']} Priority`): {t['reason']}")

            st.markdown("#### 💡 Actionable Study Tips")
            for tip in plan.get("actionable_study_tips", []):
                st.write(f"- {tip}")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 6: QUIZ & GROUNDED AI TUTOR ---
    with tab6:
        q_tab1, q_tab2 = st.tabs(["✍️ Practice Quiz", "🤖 Grounded 24/7 AI Tutor"])

        with q_tab1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='sub-gradient-header'>✍️ Active Practice Quiz</h3>", unsafe_allow_html=True)

            quizzes = st.session_state["published_quizzes"]
            if not quizzes:
                st.info("No active quiz published by the teacher right now.")
            else:
                score = 0
                with st.form("student_quiz_form"):
                    user_answers = {}
                    for idx, q in enumerate(quizzes):
                        st.markdown(f"**Q{idx+1}: {q['question']}**")
                        user_answers[q['id']] = st.radio("Select Answer:", q['options'], key=f"q_ans_{q['id']}")
                        st.markdown("---")

                    submit_quiz = st.form_submit_button("Submit Quiz Answers", type="primary")

                if submit_quiz:
                    for q in quizzes:
                        selected = user_answers.get(q['id'])
                        if selected == q['answer']:
                            score += 1
                            st.success(f"Q{q['id']}: Correct! ({q['explanation']})")
                        else:
                            st.error(f"Q{q['id']}: Incorrect. Correct answer: **{q['answer']}**. {q['explanation']}")
                    final_pct = (score / len(quizzes)) * 100.0
                    st.session_state["quiz_score_pct"] = final_pct
                    st.metric("Your Final Score", f"{score} / {len(quizzes)} ({final_pct:.0f}%)")

            st.markdown("</div>", unsafe_allow_html=True)

        with q_tab2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='sub-gradient-header'>🤖 Grounded 24/7 AI Classroom Tutor</h3>", unsafe_allow_html=True)
            st.write(f"Ask questions about today's lecture. The AI tutor responds strictly based **ONLY** on active class context in **{lang}**!")

            context_str = ""
            if st.session_state["lecture_data"]:
                context_str += st.session_state["lecture_data"].get("summary", "") + "\n" + st.session_state["lecture_data"].get("structured_notes", "")
            if st.session_state["board_data"]:
                context_str += "\n" + st.session_state["board_data"].get("extracted_text", "")

            for msg in st.session_state["tutor_chat"]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            user_query = st.chat_input(f"Ask a question in {lang}...")
            if user_query:
                st.session_state["tutor_chat"].append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("AI Tutor searching live lecture context..."):
                        ans = ai.ask_context_tutor(client, user_query, context_str, st.session_state["tutor_chat"], preferred_language=lang)
                        st.write(ans)
                        st.session_state["tutor_chat"].append({"role": "assistant", "content": ans})

            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()