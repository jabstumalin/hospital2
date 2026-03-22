"""
Federated Learning - Hospital 2 Local Dashboard
Interactive, non-linear dashboard for model synchronization, evaluation, and testing.
"""
import streamlit as st
import pandas as pd
import requests
from config import settings

st.set_page_config(
    page_title="Hospital-2 Node",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Helper Functions ---

def get_backend_artifact_status():
    try:
        response = requests.get(f"{settings.LOCAL_API_BASE_URL}/artifact-status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None

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
    
    artifact_status = get_backend_artifact_status()
    if artifact_status and artifact_status.get("model_present"):
        model_file = artifact_status.get("model_file")
        model_version = artifact_status.get("model_version")
        st.caption(f"Central model in use: {model_file} (v{model_version})")
    else:
        st.caption("Central model in use: none downloaded on backend yet")
    
    st.markdown("---")
    st.header("Danger Zone")
    if st.button("Reset All Local Models", use_container_width=True):
        try:
            response = requests.post(f"{settings.LOCAL_API_BASE_URL}/reset")
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
                response = requests.post(f"{settings.LOCAL_API_BASE_URL}/sync-global", timeout=60)
                if response.status_code == 200:
                    payload = response.json()
                    st.success(payload.get("message", "Global model package synced on backend."))
                    if not payload.get("scaler_present", False):
                        st.warning("Package synced, but global_scaler.pkl was not found on backend.")
                else:
                    detail = response.json().get("detail", "Unknown error")
                    st.error(f"Sync failed: {detail}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

artifact_status = get_backend_artifact_status()
if artifact_status and artifact_status.get("model_present"):
    st.info(
        f"Using central model: {artifact_status.get('model_file')} (v{artifact_status.get('model_version')})"
    )
else:
    st.warning("No central model found on backend. Click 'Retrieve Global Model from Server' to sync one.")

st.markdown("<br>", unsafe_allow_html=True)

# --- Performance & Training Dashboards ---
st.write("### Evaluate Model")

if st.button("Run Initial Test (75% samples)"):
    try:
        response = requests.post(f"{settings.LOCAL_API_BASE_URL}/evaluate", params={"sample_ratio": 0.75})
        if response.status_code == 200:
            st.session_state['initial_metrics'] = response.json()
        else:
            detail = response.json().get("detail", "Unknown error")
            st.error(f"Initial test failed: {detail}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the local API. Is FastAPI running?")
        
if 'initial_metrics' in st.session_state:
    st.write("**Evaluation Metrics (75% of test samples):**")
    st.table(pd.DataFrame([st.session_state['initial_metrics']]))

st.write("### Retrain Local Model")

if st.button("Retrain with Hospital-2 Data"):
    try:
        response = requests.post(f"{settings.LOCAL_API_BASE_URL}/retrain")
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