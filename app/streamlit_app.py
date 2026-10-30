"""Multimodal Voice Scam Risk Analysis Interactive Streamlit Demonstration."""

import os
import tempfile
from pathlib import Path
import streamlit as st

from voxshield.models.predict import VoxShieldPredictor
from voxshield.utils.paths import MODELS_DIR, RAW_DATA_DIR

# Set page configuration
st.set_page_config(
    page_title="Multimodal Voice Scam Risk Analysis",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Multimodal Voice Scam Risk Analysis")
st.markdown(
    "An academic research tool evaluating the complementary value of acoustic features and speech transcripts "
    "for voice phishing (vishing) risk estimation."
)

# Prominent Research Disclaimer
st.warning(
    "⚠️ **ACADEMIC RESEARCH PROTOTYPE ONLY**\n\n"
    "This tool provides an uncertainty-aware research risk score based on synthetic academic benchmark models. "
    "It is **NOT an authoritative fraud verdict** and must **NEVER** be used to execute financial freezes, account blocks, "
    "or legal accusations without certified human review."
)

# Check model artifacts
text_model_path = MODELS_DIR / "E1_tfidf_lr.joblib"
audio_model_path = MODELS_DIR / "E3_audio_lr.joblib"
fusion_model_path = MODELS_DIR / "E6_fusion_ref_meta.joblib"

if not text_model_path.exists():
    st.error(
        f"Model artifacts not found in `{MODELS_DIR}`! "
        "Please run baseline training first via `python scripts/train_baselines.py`."
    )
    st.stop()


@st.cache_resource
def load_predictor():
    return VoxShieldPredictor(
        text_model_path=text_model_path,
        audio_model_path=audio_model_path,
        fusion_model_path=fusion_model_path,
        whisper_model_name="openai/whisper-small",
    )


predictor = load_predictor()

# Sidebar: Controls & Sample audio selector
st.sidebar.header("Evaluation Controls")
threshold = st.sidebar.slider("Decision Boundary Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

st.sidebar.subheader("Select Benchmark Sample")
sample_dir = RAW_DATA_DIR / "audio"
sample_files = list(sample_dir.glob("*.wav")) if sample_dir.exists() else []

selected_sample_path = None
if sample_files:
    sample_options = ["None (Upload Custom Audio)"] + [f.name for f in sample_files[:20]]
    choice = st.sidebar.selectbox("Preloaded Audio Samples:", sample_options)
    if choice != "None (Upload Custom Audio)":
        selected_sample_path = sample_dir / choice

# Main Area: Audio Input
st.subheader("1. Audio Input")
uploaded_file = st.file_uploader("Upload Audio Recording (.wav, .mp3)", type=["wav", "mp3"])
custom_transcript = st.text_area("Optional Ground-Truth / Reference Transcript (leave blank to run Whisper ASR):")

audio_to_process = None
temp_file_path = None

if uploaded_file is not None:
    temp_dir = tempfile.mkdtemp()
    temp_file_path = Path(temp_dir) / uploaded_file.name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    audio_to_process = temp_file_path
    st.audio(uploaded_file)
elif selected_sample_path is not None and selected_sample_path.exists():
    audio_to_process = selected_sample_path
    st.info(f"Using preloaded dataset sample: `{selected_sample_path.name}`")
    with open(selected_sample_path, "rb") as f:
        st.audio(f.read())

if audio_to_process is not None:
    if st.button("Analyze Scam Risk", type="primary"):
        with st.spinner("Processing audio signals and running multimodal inference..."):
            try:
                res = predictor.predict(
                    audio_path=audio_to_process,
                    reference_transcript=custom_transcript,
                    threshold=threshold,
                )

                st.subheader("2. Model Risk Assessment")
                col1, col2, col3 = st.columns(3)

                with col1:
                    risk_pct = res["fusion_risk_score"] * 100
                    st.metric("Multimodal Risk Score", f"{risk_pct:.1f}%")
                    if res["is_scam"]:
                        st.error(f"Verdict: {res['verdict']}")
                    else:
                        st.success(f"Verdict: {res['verdict']}")

                with col2:
                    st.metric("Text-Only Risk Score", f"{res['text_risk_score'] * 100:.1f}%")
                    st.caption(f"Source: {res['transcript_source']}")

                with col3:
                    st.metric("Audio-Only Risk Score", f"{res['audio_risk_score'] * 100:.1f}%")
                    st.caption("Acoustic MFCC & Prosodic Model")

                st.subheader("3. Transcript & Lexical Evidence")
                st.text_area("Detected Transcript:", value=res["transcript"], height=120, disabled=True)
                if res["lexical_evidence"]["salient_terms"]:
                    st.write("**Top Discriminative Terms:**")
                    st.json(dict(res["lexical_evidence"]["salient_terms"]))

                st.subheader("4. Acoustic Signal Evidence")
                st.json(res["acoustic_evidence"])

            except Exception as e:
                st.error(f"Inference error: {str(e)}")
            finally:
                if temp_file_path and temp_file_path.exists():
                    try:
                        temp_file_path.unlink()
                    except Exception:
                        pass
else:
    st.info("Select a preloaded audio sample from the sidebar or upload an audio file to begin analysis.")
