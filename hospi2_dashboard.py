"""
Federated Learning - Hospital 2 Local Dashboard
Interactive, non-linear dashboard for model synchronization, evaluation, and testing.
"""
import streamlit as st
import pandas as pd
import requests
import os
import zipfile
import io
import re
from config import settings

st.set_page_config(
    page_title="Hospital-2 Node",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Helper Functions ---

def get_latest_global_model_path(model_dir: str):
    """Return (path, version) for latest main_model_vN.pkl found in model_dir."""
    latest_version = -1
    latest_path = None
    pattern = re.compile(r"^main_model_v(\d+)\.pkl$")

    if not os.path.isdir(model_dir):
        return None, None

    for name in os.listdir(model_dir):
        match = pattern.match(name)
        if match:
            version = int(match.group(1))
            if version > latest_version:
                latest_version = version
                latest_path = os.path.join(model_dir, name)

    if latest_path is None:
        return None, None

    return latest_path, latest_version

def check_server_status(url):
    try:
        response = requests.get(url, timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False

# --- Top Header & Server Status ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Hospital-2: Local Federated Node")
with col2:
    st.write("") # Spacing
    if check_server_status(settings.CENTRAL_SERVER_URL):
        st.success(f"🟢 Central Server Online\n\n{settings.CENTRAL_SERVER_URL}")
    else:
        st.error(f"🔴 Central Server Offline\n\n{settings.CENTRAL_SERVER_URL}")

st.markdown("---")

# --- Sidebar (Node Config & Danger Zone) ---
with st.sidebar:
    st.header("Node Configuration")
    st.success("Local Environment Active")
    st.info(f"Local API: http://localhost:{settings.API_PORT}")

    active_global_model_path, active_global_model_version = get_latest_global_model_path(settings.MODEL_PATH)
    if active_global_model_path:
        st.caption(f"Central model in use: {os.path.basename(active_global_model_path)} (v{active_global_model_version})")
    else:
        st.caption("Central model in use: none downloaded yet")
    
    st.markdown("---")
    st.header("Danger Zone")
    if st.button("Reset All Local Models", use_container_width=True):
        try:
            response = requests.post(f"http://127.0.0.1:{settings.API_PORT}/reset")
            if response.status_code == 200:
                data = response.json()
                # Clear session state metrics
                if 'initial_metrics' in st.session_state:
                    del st.session_state['initial_metrics']
                if 'retrained_metrics' in st.session_state:
                    del st.session_state['retrained_metrics']
                st.success("All models have been deleted!")
                if data.get("deleted_files"):
                    st.write(f"Deleted: {', '.join(data['deleted_files'])}")
                st.rerun()
            else:
                st.error("Failed to reset models via API.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the local API. Is FastAPI running?")

# --- Control Panel: Sync ---
st.subheader("Model Synchronization")
col_sync, _ = st.columns([1, 2])
with col_sync:
    if st.button("Retrieve Global Model from Server", type="primary", use_container_width=True):
        with st.spinner("Downloading global model..."):
            try:
                response = requests.get(f"{settings.CENTRAL_SERVER_URL}/global/package")
                if response.status_code == 200:
                    os.makedirs(settings.MODEL_PATH, exist_ok=True)
                    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
                    zip_file.extractall(settings.MODEL_PATH)
                    st.success("Global model successfully downloaded and stored locally!")
                else:
                    st.error(f"Failed to fetch package. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")

active_global_model_path, active_global_model_version = get_latest_global_model_path(settings.MODEL_PATH)
if active_global_model_path:
    st.info(f"Using central model: {os.path.basename(active_global_model_path)} (v{active_global_model_version})")
else:
    st.warning("No central model found locally. Click 'Retrieve Global Model from Server' to download one.")

st.markdown("<br>", unsafe_allow_html=True)

# --- Performance & Training Dashboards ---
st.write("### Evaluate Model")

if st.button("Run Initial Test (50 samples)"):
    try:
        response = requests.post(f"http://127.0.0.1:{settings.API_PORT}/evaluate", params={"sample_size": 50})
        if response.status_code == 200:
            st.session_state['initial_metrics'] = response.json()
        else:
            detail = response.json().get("detail", "Unknown error")
            st.error(f"Initial test failed: {detail}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the local API. Is FastAPI running?")
        
if 'initial_metrics' in st.session_state:
    st.write("**Evaluation Metrics (50 samples):**")
    st.table(pd.DataFrame([st.session_state['initial_metrics']]))

st.write("### Retrain Local Model")

if st.button("Retrain with Hospital-2 Data"):
    try:
        response = requests.post(f"http://127.0.0.1:{settings.API_PORT}/retrain")
        if response.status_code == 200:
            payload = response.json()
            st.success(payload.get("message", "Retraining completed."))
            st.session_state['retrained_metrics'] = payload.get("metrics", {})
        else:
            detail = response.json().get("detail", "Unknown error")
            st.error(f"Retraining failed: {detail}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the local API. Is FastAPI running?")

if 'retrained_metrics' in st.session_state:
    st.write("**Retrained Model Evaluation Metrics (All samples):**")
    st.table(pd.DataFrame([st.session_state['retrained_metrics']]))