from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from config import settings
import os
import glob
import json
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

class HealthCheck(BaseModel):
    status: str
    message: str


def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1 Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def align_features(df: pd.DataFrame, fitted_obj, obj_name: str) -> pd.DataFrame:
    expected = getattr(fitted_obj, "feature_names_in_", None)
    if expected is None:
        return df

    expected = list(expected)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns for {obj_name}: {missing}")

    return df.loc[:, expected]


def save_retrained_metrics(metrics):
    os.makedirs(settings.MODEL_PATH, exist_ok=True)
    persisted = {
        "accuracy": metrics.get("Accuracy"),
        "precision": metrics.get("Precision"),
        "recall": metrics.get("Recall"),
        "f1_score": metrics.get("F1 Score"),
    }
    with open(os.path.join(settings.MODEL_PATH, "hospital_2_v2_metrics.json"), "w") as f:
        json.dump(persisted, f)


def get_latest_global_model_path(model_dir: str):
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

@app.get("/", response_model=HealthCheck)
def root():
    return HealthCheck(status="ok", message="Hospital 2 API is running")


@app.post("/reset")
def reset_node():
    """Reset the node by deleting all locally stored models."""
    deleted_files = []
    if os.path.exists(settings.MODEL_PATH):
        for file in glob.glob(os.path.join(settings.MODEL_PATH, "*.pkl")) + glob.glob(os.path.join(settings.MODEL_PATH, "*_metrics.json")):
            try:
                os.remove(file)
                deleted_files.append(os.path.basename(file))
            except Exception as e:
                return {"status": "error", "message": str(e)}
    return {"status": "success", "message": "Node reset successfully.", "deleted_files": deleted_files}


@app.post("/evaluate")
def evaluate_model(sample_size: int = 50):
    """Run local evaluation against the latest downloaded global model."""
    active_global_model_path, active_global_model_version = get_latest_global_model_path(settings.MODEL_PATH)
    global_scaler_path = os.path.join(settings.MODEL_PATH, "global_scaler.pkl")

    if active_global_model_path is None or not os.path.exists(global_scaler_path):
        raise HTTPException(status_code=400, detail="Main model or scaler not found. Retrieve global package first.")

    try:
        model = joblib.load(active_global_model_path)
        scaler = joblib.load(global_scaler_path)

        df = pd.read_csv("dataset/set3_hosp2_test.csv")
        if "cardio" not in df.columns:
            raise HTTPException(status_code=400, detail="Test dataset must contain a 'cardio' column.")

        X = df.drop("cardio", axis=1)
        y = df["cardio"]

        safe_sample_size = min(max(sample_size, 1), len(df)) if len(df) > 0 else 0
        if safe_sample_size == 0:
            raise HTTPException(status_code=400, detail="Test dataset is empty.")

        X_sample = X.sample(n=safe_sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]

        X_sample = align_features(X_sample, scaler, "global scaler")
        X_scaled = scaler.transform(X_sample)

        preds = model.predict(X_scaled)
        metrics = calculate_metrics(y_sample, preds)
        metrics["Samples"] = safe_sample_size
        metrics["Global Model Version"] = f"v{active_global_model_version}"
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initial test failed: {e}")


@app.post("/retrain")
def retrain_model():
    """Fine-tune the latest global model on Hospital-2 data and persist artifacts."""
    active_global_model_path, active_global_model_version = get_latest_global_model_path(settings.MODEL_PATH)
    global_scaler_path = os.path.join(settings.MODEL_PATH, "global_scaler.pkl")

    if not os.path.exists(global_scaler_path) or active_global_model_path is None:
        raise HTTPException(status_code=400, detail="Global scaler or global model not found. Retrieve global package first.")

    try:
        scaler = joblib.load(global_scaler_path)
        model = joblib.load(active_global_model_path)
        train = pd.read_csv("dataset/set3_hosp2_train.csv")
        test = pd.read_csv("dataset/set3_hosp2_test.csv")

        if "cardio" not in train.columns or "cardio" not in test.columns:
            raise HTTPException(status_code=400, detail="Train and test datasets must contain a 'cardio' column.")

        X = train.drop("cardio", axis=1)
        y = train["cardio"]
        X_test = test.drop("cardio", axis=1)
        y_test = test["cardio"]

        if len(X) == 0:
            raise HTTPException(status_code=400, detail="Training dataset is empty.")
        if y.nunique() < 2:
            raise HTTPException(status_code=400, detail="Training requires at least two target classes in 'cardio'.")

        X = align_features(X, scaler, "global scaler")
        X_test = align_features(X_test, scaler, "global scaler")

        X_scaled = scaler.transform(X)
        X_test_scaled = scaler.transform(X_test)

        if not hasattr(model, "partial_fit"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "The global model does not support fine-tuning (missing 'partial_fit'). "
                    "Ensure the central server uses an SGDClassifier or similar incremental learner."
                ),
            )

        model.partial_fit(X_scaled, y, classes=np.unique(y))

        os.makedirs(settings.MODEL_PATH, exist_ok=True)
        joblib.dump(model, os.path.join(settings.MODEL_PATH, "hospital_2_v2.pkl"))
        joblib.dump(scaler, os.path.join(settings.MODEL_PATH, "local_scaler.pkl"))

        preds = model.predict(X_test_scaled)
        metrics = calculate_metrics(y_test, preds)
        save_retrained_metrics(metrics)

        return {
            "status": "success",
            "message": f"Global model v{active_global_model_version} successfully fine-tuned and saved as hospital_2_v2",
            "metrics": metrics,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

@app.get("/model/download")
def download_model():
    """Endpoint for the central server to download this node's locally retrained model."""
    # This uses the expected filename that hospi2_dashboard.py produces
    model_path = os.path.join(settings.MODEL_PATH, "hospital_2_v2.pkl")
    
    if os.path.exists(model_path):
        return FileResponse(
            path=model_path,
            filename="hospital_2_v2.pkl",
            media_type="application/octet-stream"
        )
    else:
        raise HTTPException(status_code=404, detail="Locally retrained model not found. Node must train the model first.")


@app.get("/metrics")
def get_metrics():
    """Return the evaluation metrics for the locally retrained model.

    The central server calls this endpoint to retrieve per-node metrics so it
    can persist them and compute the aggregated (FedAvg) model metrics.
    """
    metrics_path = os.path.join(settings.MODEL_PATH, "hospital_2_v2_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Metrics not available yet. Node must retrain the model first."
        )
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics

