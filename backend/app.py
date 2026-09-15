"""
VoxGuard — Live Demo Dashboard (Streamlit)
=============================================
Internal testing UI / fallback demo — kept alongside the FastAPI backend
in api/main.py, which is what the separate frontend/ talks to.

Run (from inside backend/): python -m streamlit run app.py
"""

import streamlit as st
import tempfile
import os
import time
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

from audio_layers.voxguard_layer1_synthetic_voice_detector import SyntheticVoiceDetector
import speaker_id.voxguard_layer2_speaker_verification as vg2
from context_nlp.voxguard_layer5_context_analyzer import analyze_context
from risk_engine.voxguard_risk_engine import compute_risk

st.set_page_config(page_title="VoxGuard", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_detector():
    return SyntheticVoiceDetector()

detector = load_detector()

st.title("🛡️ VoxGuard")
st.caption("Real-Time Voice Cloning & Impersonation Detection — SIH 2026 | PS 26104")

st.divider()

# ---------------------------------------------------------------------
# SECTION 1 — Enroll a known speaker
# ---------------------------------------------------------------------
st.subheader("1. Enroll a known speaker (watchlist)")
enroll_name = st.text_input("Speaker name")
enroll_file = st.file_uploader("Upload their reference voice (.wav)", type=["wav"], key="enroll")

if st.button("Enroll Speaker") and enroll_name and enroll_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(enroll_file.read())
        tmp_path = tmp.name
    vg2.enroll(enroll_name, tmp_path)
    st.success(f"Enrolled '{enroll_name}' in the watchlist.")

st.divider()

# ---------------------------------------------------------------------
# SECTION 2 — Analyze a single uploaded call
# ---------------------------------------------------------------------
st.subheader("2. Analyze an incoming call")
test_file = st.file_uploader("Upload call audio (.wav)", type=["wav"], key="test")

def render_result(final: dict, transcript: str, synth_result: dict):
    st.divider()
    st.subheader("VOICE SECURITY ANALYSIS")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Synthetic Probability", f"{final['synthetic_probability']*100:.1f}%")
        st.metric("Watchlist Match", "Yes" if final['watchlist_match'] else "No")
    with col2:
        st.metric("Intent Category", final['intent_category'])
        st.metric("Matched Speaker", final['matched_name'] or "—")

    if "num_windows_analyzed" in synth_result:
        st.caption(f"Synthetic-voice check analyzed {synth_result['num_windows_analyzed']} overlapping windows across the full clip.")

    st.divider()

    risk_color = {"HIGH-RISK CALL": "🔴", "MEDIUM-RISK CALL": "🟠", "LOW-RISK CALL": "🟢"}
    st.markdown(f"## {risk_color.get(final['risk_level'], '')} {final['risk_level']}")
    st.markdown(f"### Risk Index: {final['risk_score']}/100")
    st.caption("Fused score across voice authenticity, watchlist match, and call-intent signals — not a raw probability.")

    st.markdown("**Why:**")
    for reason in final["reasons"]:
        st.markdown(f"- {reason}")

    st.markdown("**Recommended Action:**")
    st.info(final["recommended_action"])

    with st.expander("Transcript"):
        st.write(transcript)

if st.button("Run Analysis") and test_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(test_file.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing voice, speaker identity, and call context..."):
        synth_result = detector.detect_full(tmp_path)
        watch_result = vg2.check_watchlist(tmp_path)
        context_result = analyze_context(tmp_path)
        final = compute_risk(synth_result, watch_result, context_result)

    os.unlink(tmp_path)
    render_result(final, context_result["transcript"], synth_result)

st.divider()

# ---------------------------------------------------------------------
# SECTION 3 — Live microphone monitoring
# ---------------------------------------------------------------------
st.subheader("3. Live Microphone Monitoring")
st.caption("Records short chunks from your mic and analyzes each one — simulates real-time call monitoring.")

CHUNK_SECONDS = 4
SAMPLE_RATE = 16000
MAX_CHUNKS = 15

if "live_running" not in st.session_state:
    st.session_state.live_running = False

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Start Live Monitoring"):
        st.session_state.live_running = True
with col_b:
    if st.button("Stop Live Monitoring"):
        st.session_state.live_running = False

live_placeholder = st.empty()

if st.session_state.live_running:
    for i in range(MAX_CHUNKS):
        if not st.session_state.live_running:
            break

        with live_placeholder.container():
            st.info(f"🎙️ Recording chunk {i + 1}/{MAX_CHUNKS} ({CHUNK_SECONDS}s)...")

        recording = sd.rec(int(CHUNK_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()

        chunk_path = "live_chunk.wav"
        sf.write(chunk_path, recording, SAMPLE_RATE)

        with live_placeholder.container():
            st.info("Analyzing chunk...")
            synth_result = detector.detect_full(chunk_path)
            watch_result = vg2.check_watchlist(chunk_path)
            context_result = analyze_context(chunk_path)
            final = compute_risk(synth_result, watch_result, context_result)

            risk_color = {"HIGH-RISK CALL": "🔴", "MEDIUM-RISK CALL": "🟠", "LOW-RISK CALL": "🟢"}
            st.markdown(f"## {risk_color.get(final['risk_level'], '')} {final['risk_level']} — {final['risk_score']}/100")
            st.caption(f"Transcript: {context_result['transcript']}")
            for r in final["reasons"]:
                st.write(f"- {r}")

        time.sleep(0.5)

    st.session_state.live_running = False