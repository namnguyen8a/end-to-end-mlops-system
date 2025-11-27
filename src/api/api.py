from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
import uvicorn
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Use environment variable if set, otherwise default to relative path
DATA_PATH = os.getenv("DATA_PATH", "../data/processed")

app = FastAPI(title="Insurance Weekly Price Predictor")

TICKER_RUN_IDS = {
    "BIC": "226e478c79ad48a0a6551b7d259c2f36",
    "BMI": "8a813ff6ef004ec0800e22cbd70e57c9",
    "BVH": "7d73e7505bfd4ba8944cf09392d6f4f0",
    "MIG": "7f70025b488d4afa9ad841960a3b130d",
    "PGI": "9c87b1ff438543798772971c183ec17c",
}

loaded_models = {}

class AutoPredictResponse(BaseModel):
    ticker: str
    horizon: Literal["1_week_ahead"]
    predicted_close: float
    currency: str
    generated_at_utc: str
    run_id: str
    latest_time: str

def load_model_for_ticker(ticker: str):
    if ticker not in loaded_models:
        run_id = TICKER_RUN_IDS[ticker]
        model_uri = f"runs:/{run_id}/model"
        try:
            loaded_models[ticker] = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            raise Exception(
                f"Failed to load model for {ticker} (run_id: {run_id}). "
                f"URI: {MLFLOW_TRACKING_URI}, Model URI: {model_uri}. "
                f"Error: {str(e)}. "
                f"Check if run_id has 'model/' folder in MLflow UI."
            )
    return loaded_models[ticker]

def build_latest_features(ticker: str):
    file_path = os.path.join(DATA_PATH, f"{ticker}_weekly_clean.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    df = pd.read_csv(file_path, parse_dates=["time"])
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

@app.get("/predict_next_week", response_model=AutoPredictResponse)
def predict_next_week(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKER_RUN_IDS:
        raise HTTPException(status_code=404, detail="Ticker not supported")

    try:
        features, latest_time = build_latest_features(ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not build features: {e}")

    try:
        model = load_model_for_ticker(ticker)
        predicted_close = float(model.predict([features])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    run_id = TICKER_RUN_IDS[ticker]
    return AutoPredictResponse(
        ticker=ticker,
        horizon="1_week_ahead",
        predicted_close=predicted_close,
        currency="VND",
        generated_at_utc=datetime.utcnow().isoformat() + "Z",
        run_id=run_id,
        latest_time=latest_time,
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)