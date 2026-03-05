import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
import pandas as pd
from collections import deque

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PROCTO",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif; }
code, .mono { font-family: 'IBM Plex Mono', monospace; }

[data-testid="stAppViewContainer"] {
    background: #0a0c10;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #1e2433;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #38bdf8; font-family: 'IBM Plex Mono', monospace; }
.metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

.alert-box {
    border-radius: 6px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
    border-left: 3px solid;
}
.alert-high { background: #1f0a0a; border-color: #ef4444; color: #fca5a5; }
.alert-med  { background: #1f1408; border-color: #f59e0b; color: #fcd34d; }
.alert-low  { background: #081a10; border-color: #22c55e; color: #86efac; }

.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.status-active   { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.status-warning  { background: #1c1300; color: #fbbf24; border: 1px solid #92400e; }
.status-critical { background: #1f0a0a; color: #f87171; border: 1px solid #7f1d1d; }

.log-container {
    background: #080b10;
    border: 1px solid #1e2433;
    border-radius: 6px;
    padding: 12px;
    height: 280px;
    overflow-y: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
}
.log-entry { padding: 2px 0; border-bottom: 1px solid #0d1117; color: #94a3b8; }
.log-entry span { color: #38bdf8; margin-right: 8px; }

.stButton > button {
    background: #0ea5e9 !important;
    color: #000 !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 4px !important;
}
.stButton > button:hover { background: #38bdf8 !important; }

div[data-testid="column"] > div { height: 100%; }
.stProgress > div > div { background: #0ea5e9 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "alerts" not in st.session_state:
    st.session_state.alerts = deque(maxlen=50)
if "stats" not in st.session_state:
    st.session_state.stats = {
        "frames_processed": 0,
        "faces_detected": 0,
        "violations": 0,
        "session_start": None,
    }
if "log" not in st.session_state:
    st.session_state.log = deque(maxlen=100)
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = {}  # violation_type -> last logged timestamp

# ─── Core Detection Logic ───────────────────────────────────────────────────────
@st.cache_resource
def load_detectors():
    """Load OpenCV Haar Cascade classifiers."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    return face_cascade, eye_cascade, profile_cascade

def analyze_frame(frame, cfg):
    """
    Analyze a single video frame for exam violations.
    Returns annotated frame + list of detected events.
    """
    face_cascade, eye_cascade, profile_cascade = load_detectors()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    events = []

    # ── Detect frontal faces ──────────────────────────────────────────────────
    # Histogram equalization improves detection in varied lighting
    gray_eq = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=cfg["scale_factor"],
        minNeighbors=cfg["min_neighbors"],
        minSize=(cfg["min_face_size"], cfg["min_face_size"]),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # ── Detect profile faces (head-turned) ───────────────────────────────────
    profiles = profile_cascade.detectMultiScale(gray_eq, 1.1, 6, minSize=(60, 60))

    face_count = len(faces)

    # ── Multiple-face violation ───────────────────────────────────────────────
    if face_count > 1:
        events.append(("HIGH", f"⚠ MULTIPLE FACES DETECTED ({face_count})"))

    # ── No-face violation ────────────────────────────────────────────────────
    if face_count == 0:
        if len(profiles) == 0:
            events.append(("HIGH", "⚠ NO FACE IN FRAME — subject may have left"))
        else:
            events.append(("MED", "↩ HEAD TURNED — profile detected"))

    # ── Per-face analysis ─────────────────────────────────────────────────────
    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (56, 189, 248), 2)
        cv2.putText(frame, "FACE", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1)

        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (34, 197, 94), 1)

        eye_count = len(eyes)
        if eye_count == 0 and cfg["detect_eyes_closed"]:
            events.append(("MED", "😴 EYES NOT DETECTED — possible inattention"))

        # Face-center check (should be roughly centered in frame)
        frame_cx = frame.shape[1] // 2
        face_cx  = x + w // 2
        deviation = abs(face_cx - frame_cx) / frame.shape[1]
        if deviation > cfg["gaze_threshold"]:
            events.append(("MED", f"👁 GAZE DEVIATION — {int(deviation*100)}% off-center"))

    # ── Profile annotations ───────────────────────────────────────────────────
    for (px, py, pw, ph) in profiles:
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (251, 191, 36), 2)
        cv2.putText(frame, "PROFILE", (px, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (251, 191, 36), 1)

    # ── HUD overlay ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"ExamGuard  {ts}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1)
    cv2.putText(frame, f"Faces: {face_count}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)

    status_color = (34, 197, 94) if not events else (239, 68, 68)
    cv2.circle(frame, (frame.shape[1] - 20, 20), 8, status_color, -1)

    return frame, events, face_count

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ CONFIG")
    st.markdown("---")

    cfg = {
        "scale_factor":    st.slider("Scale Factor",       1.05, 1.5,  1.2,  0.05),
        "min_neighbors":   st.slider("Min Neighbors",      1,    15,   8),
        "min_face_size":   st.slider("Min Face Size (px)", 50,   250,  100),
        "gaze_threshold":  st.slider("Gaze Sensitivity",   0.1,  0.5,  0.25, 0.05),
        "detect_eyes_closed": st.checkbox("Detect Eye Closure", False),
        "show_raw":        st.checkbox("Show Grayscale Debug", False),
        "fps_limit":       st.slider("Max FPS",            5,    30,   15),
        "cooldown":        st.slider("Alert Cooldown (sec)", 1,    30,   5),
    }

    st.markdown("---")
    st.markdown("### 📊 SESSION STATS")
    s = st.session_state.stats
    st.metric("Frames Processed", s["frames_processed"])
    st.metric("Total Violations", s["violations"])

    st.markdown("---")
    st.markdown("### ℹ️ HOW IT WORKS")
    st.markdown("""
- **Haar Cascades** detect frontal faces & eyes  
- **Profile detector** catches head turns  
- Gaze estimated from face-center deviation  
- Alerts log in real-time  
    """)

# ─── Main Layout ────────────────────────────────────────────────────────────────
st.markdown("# 🎓 ExamGuard AI")
st.markdown("<span class='mono' style='color:#64748b;font-size:.85rem'>Face-Based Exam Monitoring System · OpenCV + Streamlit</span>", unsafe_allow_html=True)
st.markdown("---")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 4])
with col_ctrl1:
    if st.button("▶  START MONITORING" if not st.session_state.monitoring else "⏹  STOP MONITORING"):
        st.session_state.monitoring = not st.session_state.monitoring
        if st.session_state.monitoring:
            st.session_state.stats["session_start"] = datetime.now()
            st.session_state.stats["frames_processed"] = 0
            st.session_state.stats["violations"] = 0
            st.session_state.alerts.clear()
            st.session_state.log.clear()
        st.rerun()

with col_ctrl2:
    if st.button("🗑  CLEAR LOG"):
        st.session_state.alerts.clear()
        st.session_state.log.clear()
        st.rerun()

# Status badge
status_html = (
    "<span class='status-badge status-active'>● MONITORING ACTIVE</span>"
    if st.session_state.monitoring
    else "<span class='status-badge' style='background:#111;color:#475569;border:1px solid #1e2433'>○ IDLE</span>"
)
st.markdown(status_html, unsafe_allow_html=True)
st.markdown("")

# ─── Video + Metrics ────────────────────────────────────────────────────────────
col_vid, col_right = st.columns([3, 2])

with col_vid:
    st.markdown("### 📹 LIVE FEED")
    video_placeholder = st.empty()

with col_right:
    st.markdown("### 📈 LIVE METRICS")
    m1, m2, m3 = st.columns(3)
    faces_ph    = m1.empty()
    violations_ph = m2.empty()
    fps_ph      = m3.empty()

    st.markdown("")
    st.markdown("### 🚨 ALERTS")
    alerts_ph = st.empty()

st.markdown("---")
st.markdown("### 📋 EVENT LOG")
log_ph = st.empty()

# ─── Monitoring Loop ────────────────────────────────────────────────────────────
if st.session_state.monitoring:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open webcam. Make sure it is connected and not in use by another app.")
        st.session_state.monitoring = False
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_delay = 1.0 / cfg["fps_limit"]
    prev_time   = time.time()

    try:
        while st.session_state.monitoring:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Frame capture failed.")
                break

            now = time.time()
            elapsed = now - prev_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
            fps_actual = 1.0 / max(time.time() - prev_time, 1e-6)
            prev_time  = time.time()

            annotated, events, face_count = analyze_frame(frame.copy(), cfg)

            if cfg["show_raw"]:
                gray_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                annotated = cv2.cvtColor(gray_disp, cv2.COLOR_GRAY2BGR)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", width="stretch")

            # Update stats
            st.session_state.stats["frames_processed"] += 1
            st.session_state.stats["faces_detected"] = face_count
            # Log events — with per-type cooldown to avoid spam
            ts = datetime.now().strftime("%H:%M:%S")
            now_t = time.time()
            cooldown = cfg["cooldown"]
            for sev, msg in events:
                # Use first 30 chars of message as the key (ignores dynamic numbers)
                key = msg[:30]
                last = st.session_state.last_alert_time.get(key, 0)
                if now_t - last >= cooldown:
                    st.session_state.last_alert_time[key] = now_t
                    st.session_state.alerts.appendleft({"time": ts, "severity": sev, "msg": msg})
                    st.session_state.log.appendleft(f"[{ts}] [{sev}] {msg}")
                    st.session_state.stats["violations"] += 1

            # ── Metrics ──────────────────────────────────────────────────────
            faces_ph.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{face_count}</div>
                <div class='metric-label'>Faces</div>
            </div>""", unsafe_allow_html=True)

            violations_ph.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#ef4444'>{st.session_state.stats['violations']}</div>
                <div class='metric-label'>Violations</div>
            </div>""", unsafe_allow_html=True)

            fps_ph.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#a78bfa'>{fps_actual:.0f}</div>
                <div class='metric-label'>FPS</div>
            </div>""", unsafe_allow_html=True)

            # ── Alerts panel ─────────────────────────────────────────────────
            alerts_html = ""
            for a in list(st.session_state.alerts)[:10]:
                css = "alert-high" if a["severity"] == "HIGH" else ("alert-med" if a["severity"] == "MED" else "alert-low")
                alerts_html += f"<div class='alert-box {css}'><b>{a['time']}</b>  {a['msg']}</div>"
            if not alerts_html:
                alerts_html = "<div style='color:#475569;font-size:.8rem;padding:8px'>No alerts yet...</div>"
            alerts_ph.markdown(alerts_html, unsafe_allow_html=True)

            # ── Log ───────────────────────────────────────────────────────────
            log_entries = "".join(
                f"<div class='log-entry'><span>{e.split(']')[0][1:]}</span>{']'.join(e.split(']')[1:])}</div>"
                for e in list(st.session_state.log)[:40]
            )
            log_ph.markdown(f"<div class='log-container'>{log_entries or '<span style=color:#334155>Awaiting events...</span>'}</div>", unsafe_allow_html=True)

    finally:
        cap.release()

else:
    # ── Idle state ────────────────────────────────────────────────────────────
    video_placeholder.markdown("""
    <div style='background:#080b10;border:1px dashed #1e2433;border-radius:8px;
                height:360px;display:flex;align-items:center;justify-content:center;
                flex-direction:column;gap:12px;'>
        <div style='font-size:3rem'>🎥</div>
        <div style='color:#334155;font-family:"IBM Plex Mono",monospace;font-size:.9rem'>
            Press START MONITORING to begin
        </div>
    </div>""", unsafe_allow_html=True)

    faces_ph.markdown("<div class='metric-card'><div class='metric-value'>—</div><div class='metric-label'>Faces</div></div>", unsafe_allow_html=True)
    violations_ph.markdown("<div class='metric-card'><div class='metric-value' style='color:#ef4444'>—</div><div class='metric-label'>Violations</div></div>", unsafe_allow_html=True)
    fps_ph.markdown("<div class='metric-card'><div class='metric-value' style='color:#a78bfa'>—</div><div class='metric-label'>FPS</div></div>", unsafe_allow_html=True)

    alerts_ph.markdown("<div style='color:#475569;font-size:.8rem;padding:8px'>Start monitoring to see alerts.</div>", unsafe_allow_html=True)
    log_ph.markdown("<div class='log-container'><span style='color:#334155'>Awaiting session start...</span></div>", unsafe_allow_html=True)