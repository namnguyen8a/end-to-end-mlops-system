from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Dict
import mlflow
import pandas as pd
from datetime import datetime
import uvicorn
import os
import io

from google.cloud import storage  

# -------------------------------------------------------------------
# MLflow configuration
# -------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# GCS location for processed data, e.g.
# gs://mlops-tiker-bucket/processed
GCS_PROCESSED_PREFIX = os.getenv(
    "GCS_PROCESSED_PREFIX",
    "gs://mlops-tiker-bucket/processed",
)

# FastAPI app
app = FastAPI(title="Insurance Weekly Price Predictor")

# One registry model per ticker
MODEL_NAMES: Dict[str, str] = {
    "BIC": "BIC_weekly_linear",
    "BMI": "BMI_weekly_linear",
    "BVH": "BVH_weekly_linear",
    "MIG": "MIG_weekly_linear",
    "PGI": "PGI_weekly_linear",
}

DEFAULT_STAGE = os.getenv("MODEL_STAGE", "Staging")

loaded_models: Dict[str, object] = {}
loaded_uris: Dict[str, str] = {}

# GCS client (uses GOOGLE_APPLICATION_CREDENTIALS in container)
storage_client = storage.Client()


class AutoPredictResponse(BaseModel):
    ticker: str
    horizon: Literal["1_week_ahead"]
    predicted_close: float
    currency: str
    generated_at_utc: str
    model_uri: str
    latest_time: str


def _read_processed_csv_from_gcs(ticker: str) -> pd.DataFrame:
    """
    Read {ticker}_weekly_clean.csv from GCS, under GCS_PROCESSED_PREFIX.
    Example prefix: gs://mlops-tiker-bucket/processed
    """
    if not GCS_PROCESSED_PREFIX.startswith("gs://"):
        raise ValueError("GCS_PROCESSED_PREFIX must start with 'gs://'")

    # Split "gs://bucket/path" into bucket + prefix
    _, path = GCS_PROCESSED_PREFIX.split("gs://", 1)
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    blob_path = f"{prefix}/{ticker}_weekly_clean.csv" if prefix else f"{ticker}_weekly_clean.csv"

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"Blob gs://{bucket_name}/{blob_path} not found")

    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data), parse_dates=["time"])


def load_model_for_ticker(ticker: str):
    """
    Load the MLflow model for a given ticker from the Model Registry,
    using the configured stage (default: 'Staging').
    Caches the model in memory for reuse.
    """
    if ticker not in MODEL_NAMES:
        raise HTTPException(status_code=404, detail="Ticker not supported")

    if ticker not in loaded_models:
        model_name = MODEL_NAMES[ticker]
        model_uri = f"models:/{model_name}/{DEFAULT_STAGE}"
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Failed to load model for {ticker} from registry. "
                    f"Tracking URI: {MLFLOW_TRACKING_URI}, Model URI: {model_uri}. "
                    f"Error: {str(e)}. "
                    f"Check that model '{model_name}' has a version in stage "
                    f"'{DEFAULT_STAGE}' and that artifacts are accessible."
                ),
            )
        loaded_models[ticker] = model
        loaded_uris[ticker] = model_uri

    return loaded_models[ticker]


def build_latest_features(ticker: str):
    """
    Load the latest processed weekly data for a ticker from GCS and construct
    the feature vector expected by the model.
    """
    df = _read_processed_csv_from_gcs(ticker)
    df = df.sort_values("time").reset_index(drop=True)

    df["close_lag1"] = df["close"].shift(1)
    df["close_lag2"] = df["close"].shift(2)
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["std_5"] = df["close"].rolling(window=5).std()
    df = df.dropna().reset_index(drop=True)

    latest_row = df.iloc[-1]
    features = [
        float(latest_row["close_lag1"]),
        float(latest_row["close_lag2"]),
        float(latest_row["ma_5"]),
        float(latest_row["ma_10"]),
        float(latest_row["std_5"]),
    ]
    latest_time = latest_row["time"].isoformat()
    return features, latest_time



@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "stage": DEFAULT_STAGE,
        "gcs_prefix": GCS_PROCESSED_PREFIX,
    }


@app.get("/predict_next_week", response_model=AutoPredictResponse)
def predict_next_week(ticker: str):
    ticker = ticker.upper()
    if ticker not in MODEL_NAMES:
        raise HTTPException(status_code=404, detail="Ticker not supported")

    # Build features from latest data
    try:
        features, latest_time = build_latest_features(ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not build features: {e}")

    # Load registry model and predict
    try:
        model = load_model_for_ticker(ticker)
        predicted_close = float(model.predict([features])[0])
        model_uri = loaded_uris[ticker]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return AutoPredictResponse(
        ticker=ticker,
        horizon="1_week_ahead",
        predicted_close=predicted_close,
        currency="VND",
        generated_at_utc=datetime.utcnow().isoformat() + "Z",
        model_uri=model_uri,
        latest_time=latest_time,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
