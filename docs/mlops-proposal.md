# 📊 MLOps Capstone: Vietnam Stock Price Prediction Assistant
## Complete Technical Proposal & Implementation Plan

**Project Date:** November 2025  
**Target Deployment:** Colab Free + FastAPI + Docker Compose  
**Status:** Ready to Code ✅

---

## 🎯 EXECUTIVE SUMMARY

### Problem Statement (Market Research)
- **Target Users:** Vietnamese retail investors (F0/F1/F2) who trade on HoSE/HNX
- **Pain Points:**
  - Cannot process massive financial data manually
  - Need simple, actionable price predictions monthly
  - Want pattern recognition (up/down/sideways) not just point forecasts
  - Prefer conversational interface (chatbot) over dashboards
  - Trust models that combine ML + Deep Learning perspectives

### Solution: "**Stock Vision - AI Assistant cho Nhà Đầu Tư Việt Nam**"
A **production-ready MLOps pipeline** that:
- ✅ Predicts **monthly stock price movements** (not daily noise)
- ✅ Combines **Linear Regression** (price targets) + **LSTM/GRU** (pattern recognition)
- ✅ Exposes via **Chatbot UX** (Flask/Gradio frontend)
- ✅ Demonstrates **full MLOps** (Airflow DAGs, MLflow registry, Prometheus monitoring)
- ✅ Deployable on **Colab Free + Docker Compose** (no GPU needed for inference)
- ✅ Uses **public APIs** (vnstock, FiinGroup) - no scraping

---

## 📈 WHY THIS USE CASE? (vs. Options A–D)

| Criteria | Option A | Option B | Option C | **Our Choice** |
|----------|----------|----------|----------|---|
| **User Value** | Short-term risk alerts | Company health scores | Value/Growth screening | Monthly price predictions |
| **Data Availability** | APIs exist | APIs exist | APIs exist | ✅ **Best** |
| **ML Model Fit** | Classification | Classification | Classification | **Regression + Classification** |
| **Chatbot UX** | Good | Good | Good | ✅ **Most Natural** |
| **MLOps Showcase** | Moderate | Moderate | Moderate | ✅ **Ideal** |
| **Market Demand** | Very High | High | High | ✅ **Highest** |
| **Complexity/Value** | Low complexity | Medium | Medium | ✅ **High (balanced)** |
| **Feature Engineering** | Limited | Limited | Limited | ✅ **Rich** |

**Why Monthly Price Prediction wins:**
1. **Technical indicators are richer**: Momentum (4w, 12w, 26w), MA crossovers, volatility, volume, RSI, MACD
2. **Solves real problem**: Investors want to know "should I buy NOW? What's my target price next month?"
3. **Perfect for ensemble**: Linear models capture trends, LSTM captures nonlinear patterns
4. **Natural chatbot dialogue**:
   - User: "Dự đoán giá VNM tháng sau?"
   - Bot: "Giá target: 84.5k (±5%), Pattern: Tăng, Confidence: 78%"
5. **Full MLOps pipeline**: Train, registry, auto-deploy, monitoring all make sense
6. **Practical constraints**: No GPU needed (LSTM can be small), fits Colab Free

---

## 🏗️ ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  VNStock API → vnstock Python lib → PostgreSQL (raw data)   │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                  ORCHESTRATION (Airflow)                    │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │ Fetch OHLCV  │ Feature Eng  │ Label Gen    │             │
│  └──────────────┴──────────────┴──────────────┘             │
│  DAG runs: Weekly (Monday 00:00)                            │
│  Versioning: DVC tracks feature & raw data                  │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│              TRAINING (MLflow + Colab)                      │
│  ┌─────────────────────┬─────────────────────┐              │
│  │ Linear Regression   │ LSTM Classifier     │              │
│  │ (Price Target)      │ (Pattern: Up/Dn)    │              │
│  │ MAE: <2%            │ Accuracy: >72%      │              │
│  └─────────────────────┴─────────────────────┘              │
│  MLflow logs: params, metrics, artifacts                    │
│  Model Registry: Auto-promote if metric > threshold         │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│           INFERENCE (FastAPI + Docker)                      │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │ Load 2 models│ Feature cache│ Batch predict│             │
│  │ from registry│ (Redis)      │ 30+ stocks   │             │
│  └──────────────┴──────────────┴──────────────┘             │
│  Endpoint: POST /predict/{symbol}                           │
│  Response: {price_target, pattern, confidence}             │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│           FRONTEND (Chatbot + Gradio)                       │
│  ┌──────────────┬──────────────┐                            │
│  │ Gradio UI    │ FastAPI calls│                            │
│  │ Chat history │ + LLM format │                            │
│  └──────────────┴──────────────┘                            │
│  User: "Cho tôi giá VNM tháng sau"                          │
│  Bot: [calls /predict] + "Dự đoán: 84.5k, Tăng, 78% TĐ"    │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│        MONITORING (Prometheus + Grafana)                    │
│  Metrics: Model latency, batch prediction time              │
│  Alerts: Prediction error drift, API downtime               │
│  Logs: All predictions stored for backtest analysis         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 DATA PIPELINE ARCHITECTURE

### Data Sources (Real-time API)

**Primary:** VNStock (free, public)
```python
# Key endpoints:
- Quote.history(symbol, start, end) → OHLCV daily
- Available symbols: 700+ VN stocks (HoSE + HNX)
- Rate limit: ~100 req/min (free tier OK)
- Data delay: 1 day (T+1)

# Alternative: FiinGroup FiinQuant (premium, but free trial)
- Real-time tick data, smart money flow
```

### Feature Engineering (20 Weekly Indicators)

| Category | Features | Source | Use Case |
|----------|----------|--------|----------|
| **Momentum** | ret_1w, ret_4w, ret_12w, ret_26w, ret_52w | Daily OHLCV | Core alpha |
| **Trend** | MA_cross_10_40, price_vs_MA20, breakout_20w | Daily Close | Regime detection |
| **Volatility** | vol_4w, ATR_4w_pct, MAX_1w, bb_width_w | Daily OHLC | Risk filter |
| **Volume** | vol_zscore_12w, turnover_w, obv_mom_4w | Daily Volume | Conviction |
| **Liquidity** | amihud_w, illiq_rank | Daily trades | Execution risk |
| **Oscillators** | RSI_14w, RSI_speed, MACD_hist_w | Weekly Close | Non-linear signals |

**Calculation Pipeline:**
```
Daily OHLCV → Resample Weekly → Compute 20 Indicators → NaN fill → Normalize → Feature matrix
```

### Data Schema (PostgreSQL)

```sql
-- Raw Data
TABLE raw_ohlcv (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    date DATE,
    open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume BIGINT,
    created_at TIMESTAMP,
    UNIQUE(symbol, date)
);

-- Engineered Features
TABLE features_weekly (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    date DATE,  -- week end
    ret_1w FLOAT, ret_4w FLOAT, ret_12w FLOAT, ...,
    MACD_hist_w FLOAT,
    created_at TIMESTAMP,
    UNIQUE(symbol, date)
);

-- Labels (Target)
TABLE labels_monthly (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    date_start DATE,  -- month start
    price_target FLOAT,  -- next month close
    pattern_label VARCHAR(10),  -- 'UP', 'DOWN', 'SIDEWAYS'
    confidence FLOAT,  -- ground truth: actual return next month
    created_at TIMESTAMP
);

-- Model Predictions
TABLE predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    prediction_date DATE,
    price_target FLOAT,
    pattern_pred VARCHAR(10),
    confidence FLOAT,
    model_version VARCHAR(20),
    created_at TIMESTAMP
);
```

---

## 🔄 AIRFLOW DAG STRUCTURE

### DAG: `stock_price_prediction_pipeline`

**Frequency:** Weekly (Monday 00:00 UTC+7)  
**SLA:** 4 hours  
**Retry:** 3 times (exponential backoff)

```
┌──────────────────┐
│  start_pipeline  │
└────────┬─────────┘
         │
┌────────▼──────────────────────────────────────────┐
│ fetch_data_task                                    │
│  • For each symbol in UNIVERSE (top 50 HoSE)      │
│  • Call vnstock.Quote.history(last 3 years)       │
│  • Insert into raw_ohlcv table                     │
│  Duration: ~2-3 min, Depends: none                │
└────────┬──────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────┐
│ feature_engineering_task                           │
│  • Load raw_ohlcv (last 2 years)                   │
│  • Compute 20 weekly indicators (pandas)           │
│  • Normalize: StandardScaler per symbol            │
│  • Handle NaN: forward fill + drop first 10 weeks  │
│  • Save to features_weekly table                   │
│  Duration: ~1-2 min, Depends: fetch_data_task     │
└────────┬──────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────┐
│ label_generation_task                              │
│  • For each symbol, compute next month's target    │
│  • Label pattern: UP (>1%), DOWN (<-1%), SIDEWAYS  │
│  • Store in labels_monthly table                   │
│  Duration: ~30 sec, Depends: feature_engineering  │
└────────┬──────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────┐
│ data_validation_task                               │
│  • Check NaN%, feature dist, label balance         │
│  • Alert if any symbol has <50 weeks of data       │
│  Duration: ~1 min, Depends: label_generation      │
└────────┬──────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────┐
│ dvc_commit_task                                    │
│  • DVC add features_weekly/                        │
│  • Git commit with message: "DAG run {run_date}"   │
│  Duration: ~1 min, Depends: data_validation       │
└────────┬──────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────┐
│ training_trigger_task                              │
│  • Trigger Colab notebook via API (optional)       │
│  • Or: skip if running locally                     │
│  Duration: <1 sec, Depends: dvc_commit_task       │
└────────┬──────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────┐
│ end_pipeline                                       │
└────────────────────────────────────────────────────┘
```

**DAG Code Snippet:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_eng',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_price_prediction_pipeline',
    default_args=default_args,
    description='Weekly stock data + feature eng pipeline',
    schedule_interval='0 0 * * 1',  # Monday 00:00
    catchup=False,
    start_date=datetime(2025, 11, 20),
)

fetch_task = PythonOperator(
    task_id='fetch_data_task',
    python_callable=fetch_vnstock_data,
    op_kwargs={'symbols': TOP_50_STOCKS},
    dag=dag,
)

feature_task = PythonOperator(
    task_id='feature_engineering_task',
    python_callable=compute_features,
    dag=dag,
    depends_on_past=False,
)

label_task = PythonOperator(
    task_id='label_generation_task',
    python_callable=generate_labels,
    dag=dag,
    depends_on_past=False,
)

# ... more tasks

fetch_task >> feature_task >> label_task >> [validation_task, dvc_task]
```

---

## 🧠 ML MODEL ARCHITECTURE

### Model 1: Linear Regression (Price Target)

**Task:** Continuous prediction of next month's closing price  
**Input:** 20 normalized weekly features  
**Output:** price_target (continuous value)  
**Loss:** MAE (robust to outliers)

```python
class PriceTargetModel:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', Ridge(alpha=1.0, solver='auto')),
        ])
    
    def train(self, X, y):
        """
        X: (n_samples, 20) normalized features
        y: (n_samples,) next month's closing price
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Returns: price_target"""
        return self.model.predict(X)
```

**Hyperparameters:**
- Polynomial degree: 2 (captures interaction effects)
- Ridge alpha: 1.0 (light regularization)
- Test size: 20% (last 10 weeks for temporal validation)

**Expected Performance:**
- MAE: 1.5–2.5% of price
- R²: 0.55–0.70

### Model 2: LSTM for Pattern Classification

**Task:** Classify next month's price movement pattern  
**Classes:** UP (≥1%), DOWN (≤-1%), SIDEWAYS ([-1%, 1%])  
**Architecture:** Stacked LSTM with attention

```python
class PatternLSTMModel:
    def __init__(self, seq_len=8, n_features=20):
        """
        seq_len: 8 weeks of lookback
        n_features: 20 technical indicators
        """
        self.model = keras.Sequential([
            layers.LSTM(64, activation='relu', return_sequences=True, 
                       input_shape=(seq_len, n_features)),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu', return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax'),  # UP, DOWN, SIDEWAYS
        ])
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
    
    def train(self, X_seq, y_labels, epochs=50, batch_size=32):
        """
        X_seq: (n_samples, 8, 20) sequential features
        y_labels: (n_samples, 3) one-hot encoded
        """
        history = self.model.fit(
            X_seq, y_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True
                ),
            ]
        )
        return history
    
    def predict(self, X_seq):
        """Returns: probabilities for [UP, DOWN, SIDEWAYS]"""
        return self.model.predict(X_seq)
```

**Architecture Rationale:**
- **Sequence length 8 weeks**: Captures 2-month trend (avoiding noise, capturing seasonality)
- **2-layer LSTM**: Balance between capacity & overfitting risk
- **Dropout 0.2**: Light regularization for small dataset
- **Softmax output**: Naturally produces confidence scores

**Expected Performance:**
- Accuracy: 65–72%
- Precision (UP): 70–75%
- Recall (UP): 60–70%
- F1-score: 0.65–0.72

### Ensemble Inference

**How they work together:**

```python
def ensemble_predict(symbol, date):
    """
    1. Linear model predicts: price_target (e.g., 84.5)
    2. LSTM model predicts: pattern probs [P_up, P_down, P_sideways]
    3. Combine into actionable insight
    """
    
    # Load latest features for symbol
    features = fetch_latest_features(symbol)
    
    # Model 1: Price target
    price_target = lr_model.predict(features)
    
    # Model 2: Pattern
    X_seq = prepare_sequences(features)  # (1, 8, 20)
    pattern_probs = lstm_model.predict(X_seq)
    pattern_label = ['UP', 'DOWN', 'SIDEWAYS'][np.argmax(pattern_probs)]
    confidence = np.max(pattern_probs)
    
    # Return ensemble prediction
    return {
        'symbol': symbol,
        'price_target': price_target,
        'pattern': pattern_label,
        'confidence': confidence,
        'pattern_probs': {
            'up': pattern_probs[0],
            'down': pattern_probs[1],
            'sideways': pattern_probs[2],
        }
    }
```

---

## 🎯 MLflow Model Registry & Auto-Deployment

### Registry Setup

**MLflow Server:** Running locally (Docker) or on Colab  
**Artifact Store:** S3 or Google Cloud Storage  
**Backend Store:** PostgreSQL

```yaml
# mlflow/backend-store-uri: postgresql://user:pass@localhost/mlflow
# mlflow/artifact-store-uri: s3://my-bucket/mlflow-artifacts
```

### Model Registry Structure

```
Models:
├── price_target_model
│   ├── version 1
│   │   └── stage: Archived
│   ├── version 2
│   │   └── stage: Staging
│   └── version 3
│       └── stage: Production
│           └── Metric: MAE = 1.8%
│
├── pattern_lstm_model
│   ├── version 1
│   │   └── stage: Archived
│   └── version 2
│       └── stage: Production
│           └── Metric: Accuracy = 71%
```

### Auto-Promotion Rules

**Price Target Model:**
- Promote if: `MAE_new < MAE_prod - 0.2%`
- Monitor: Prediction error drift (rolling 100-point window)

**Pattern LSTM:**
- Promote if: `Accuracy_new > Accuracy_prod + 1%`
- Monitor: Class imbalance, prediction distribution shift

### Promotion Workflow

```python
def evaluate_and_promote(new_model_uri, model_name):
    """
    1. Load production model metrics
    2. Evaluate new model on holdout test set
    3. If better → register as Staging
    4. If passes monitoring → promote to Production
    """
    
    prod_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    prod_metrics = get_model_metrics(model_name, "Production")
    
    new_model = mlflow.pyfunc.load_model(new_model_uri)
    new_metrics = evaluate_model(new_model, test_set)
    
    if is_better(new_metrics, prod_metrics):
        mlflow.register_model(new_model_uri, model_name)
        mlflow.transition_model_version_stage(
            model_name=model_name,
            version=new_version,
            stage="Staging"
        )
        print(f"✅ {model_name} v{new_version} promoted to Staging")
        # Later: manual approval to Production
```

---

## 🚀 FastAPI Inference Server

### Endpoints

#### 1. Single Stock Prediction
```http
POST /predict/{symbol}
Content-Type: application/json

Response:
{
  "symbol": "VNM",
  "prediction_date": "2025-11-20",
  "price_target": 84.5,
  "price_target_confidence": 0.68,
  "pattern": "UP",
  "pattern_confidence": 0.78,
  "pattern_probs": {
    "up": 0.78,
    "down": 0.15,
    "sideways": 0.07
  },
  "model_versions": {
    "price_model": 3,
    "pattern_model": 2
  }
}
```

#### 2. Batch Prediction (Top 30 Stocks)
```http
POST /predict/batch
Content-Type: application/json

Response:
{
  "predictions": [
    { "symbol": "VNM", ... },
    { "symbol": "VCB", ... },
    ...
  ],
  "batch_id": "batch_2025-11-20_001",
  "processed_at": "2025-11-20T10:30:45Z"
}
```

#### 3. Model Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "models": {
    "price_target": {
      "version": 3,
      "stage": "Production",
      "last_loaded": "2025-11-20T08:00:00Z"
    },
    "pattern_lstm": {
      "version": 2,
      "stage": "Production",
      "last_loaded": "2025-11-20T08:00:00Z"
    }
  },
  "inference_latency_ms": 45
}
```

### FastAPI Implementation

```python
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

app = FastAPI(title="Stock Vision API", version="1.0.0")

# Load models at startup
price_model = None
pattern_model = None

@app.on_event("startup")
def load_models():
    global price_model, pattern_model
    price_model = mlflow.pyfunc.load_model("models:/price_target_model/Production")
    pattern_model = mlflow.pyfunc.load_model("models:/pattern_lstm_model/Production")

@app.post("/predict/{symbol}")
def predict(symbol: str):
    """Single stock prediction"""
    try:
        # Fetch latest features from DB
        features = fetch_features(symbol)
        
        # Predict
        price = price_model.predict(features)
        pattern_probs = pattern_lstm.predict(features)
        
        return JSONResponse({
            "symbol": symbol,
            "price_target": float(price[0]),
            "pattern": ["UP", "DOWN", "SIDEWAYS"][np.argmax(pattern_probs)],
            "confidence": float(np.max(pattern_probs)),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/predict/batch")
def predict_batch():
    """Batch prediction for top 30 stocks"""
    results = []
    for symbol in TOP_30_STOCKS:
        results.append(predict(symbol))
    return {"predictions": results}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": price_model is not None and pattern_model is not None
    }
```

### Docker Setup (FastAPI)

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 💬 CHATBOT FRONTEND (Gradio)

### Gradio Interface

```python
import gradio as gr
import requests

API_URL = "http://localhost:8000"

def chat_with_bot(user_message, history):
    """
    Conversational interface for stock predictions
    """
    
    # Extract symbol from message (simple NER)
    symbols = extract_symbols(user_message)
    
    if not symbols:
        return history + [(
            user_message,
            "❌ Vui lòng nhập mã cổ phiếu (ví dụ: VNM, VCB, FPT)"
        )]
    
    predictions = []
    for symbol in symbols:
        resp = requests.post(f"{API_URL}/predict/{symbol}").json()
        predictions.append(resp)
    
    # Format response
    bot_response = format_prediction_response(predictions)
    
    return history + [(user_message, bot_response)]

def format_prediction_response(predictions):
    """Format predictions as natural language"""
    result = "📊 **Dự Đoán Giá Tháng Sau:**\n\n"
    
    for pred in predictions:
        symbol = pred['symbol']
        price = pred['price_target']
        pattern = pred['pattern']
        conf = pred['confidence']
        
        # Emoji mapping
        emoji = "📈" if pattern == "UP" else "📉" if pattern == "DOWN" else "↔️"
        
        result += f"**{symbol}**\n"
        result += f"  {emoji} Dự đoán: **{pattern}**\n"
        result += f"  💰 Giá target: **{price:.1f}K** (TĐ: {conf:.0%})\n"
        result += f"  Xác suất: ↑{pred['pattern_probs']['up']:.0%} ↓{pred['pattern_probs']['down']:.0%} ↔️{pred['pattern_probs']['sideways']:.0%}\n\n"
    
    return result

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Stock Vision - AI Advisor cho Nhà Đầu Tư Việt Nam")
    
    with gr.Row():
        gr.Markdown("""
        Hỏi bot để nhận **dự đoán giá cổ phiếu tháng sau** + **phân tích xu hướng**
        
        Ví dụ:
        - "Giá VNM tháng sau là bao nhiêu?"
        - "Dự đoán VCB, VCI, FPT"
        - "Mã nào tăng lên tháng tới?"
        """)
    
    chatbot = gr.Chatbot(
        label="Chat History",
        scale=1,
        height=400,
        bubble_full_width=False
    )
    
    with gr.Row():
        user_input = gr.Textbox(
            label="Your Question",
            placeholder="Nhập mã cổ phiếu (ví dụ: VNM) hoặc câu hỏi...",
            scale=4
        )
        submit_btn = gr.Button("Ask", scale=1)
    
    submit_btn.click(
        fn=chat_with_bot,
        inputs=[user_input, chatbot],
        outputs=chatbot
    )
    
    user_input.submit(
        fn=chat_with_bot,
        inputs=[user_input, chatbot],
        outputs=chatbot
    )

demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
```

### Docker Setup (Gradio)

```dockerfile
# Dockerfile.gradio
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY chatbot.py ./
COPY config/ ./config/

CMD ["python", "chatbot.py"]
```

---

## 📊 MONITORING (Prometheus + Grafana)

### Metrics to Track

**Model Metrics:**
- Prediction accuracy drift (weekly)
- MAE for price predictions
- Classification metrics (precision, recall, F1 per class)
- Feature correlation changes

**Infrastructure Metrics:**
- API endpoint latency (p50, p95, p99)
- Batch prediction throughput (stocks/sec)
- Error rates (4xx, 5xx)
- Model inference time

**Business Metrics:**
- Number of predictions served (daily)
- Chatbot conversation volume
- User retention (weekly)

### Prometheus Setup

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fastapi_api'
    static_configs:
      - targets: ['localhost:8000']
    
  - job_name: 'mlflow_server'
    static_configs:
      - targets: ['localhost:5000']

  - job_name: 'airflow_scheduler'
    static_configs:
      - targets: ['localhost:8794']
```

### Grafana Dashboard Panels

**Panel 1: Model Performance Over Time**
- X: Date (weekly)
- Y: MAE (price model) + Accuracy (LSTM model)
- Threshold line: SLA threshold

**Panel 2: API Latency Distribution**
- Percentiles: p50, p95, p99
- Alert: if p95 > 100ms

**Panel 3: Prediction Volume**
- Stacked area: Batch predictions, single predictions
- Daily total

**Panel 4: Error Rate**
- Line: Daily 5xx error %
- Alert: if > 1%

---

## 🐳 DOCKER COMPOSE ORCHESTRATION

### docker-compose.yml

```yaml
version: '3.8'

services:
  # PostgreSQL (Data Store)
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: stock_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: stock_ml
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U stock_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # MLflow Server
  mlflow:
    image: python:3.10-slim
    working_dir: /app
    command: >
      sh -c "pip install mlflow psycopg2-binary &&
             mlflow server --host 0.0.0.0 --port 5000
             --backend-store-uri postgresql://stock_user:${DB_PASSWORD}@postgres:5432/stock_ml
             --default-artifact-root /mlflow/artifacts"
    ports:
      - "5000:5000"
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      MLFLOW_TRACKING_URI: postgresql://stock_user:${DB_PASSWORD}@postgres:5432/stock_ml

  # Airflow Scheduler & Webserver
  airflow:
    image: apache/airflow:2.7.1
    environment:
      AIRFLOW_HOME: /airflow
      AIRFLOW__CORE__DAGS_FOLDER: /airflow/dags
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql://stock_user:${DB_PASSWORD}@postgres:5432/airflow
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    ports:
      - "8080:8080"
    volumes:
      - ./dags:/airflow/dags
      - ./logs:/airflow/logs
    depends_on:
      postgres:
        condition: service_healthy
    command: >
      bash -c "airflow db init &&
               airflow webserver -p 8080 &
               airflow scheduler"

  # FastAPI Inference Server
  api:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      DATABASE_URL: postgresql://stock_user:${DB_PASSWORD}@postgres:5432/stock_ml
      MODEL_REGISTRY: mlflow
    depends_on:
      - postgres
      - mlflow
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Gradio Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.gradio
    ports:
      - "7860:7860"
    environment:
      API_URL: http://api:8000
    depends_on:
      - api
    command: python chatbot.py

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: 'false'
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  postgres_data:
  mlflow_artifacts:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

### .env File

```bash
# Database
DB_PASSWORD=secure_password_here

# Colab Integration (optional)
COLAB_NOTEBOOK_ID=YOUR_NOTEBOOK_ID

# Stock Universe
TOP_30_STOCKS=VNM,VCB,ACB,VIC,TCB,CTG,BID,VPB,MWG,FPT,SAB,GMD,MSN,HPG,NVL,FRT,GVR,OGC,AAA,PNJ

# API Rate Limits
VNSTOCK_API_RATE_LIMIT=100

# Model Thresholds
PRICE_MAE_THRESHOLD=2.5
PATTERN_ACCURACY_THRESHOLD=70
```

---

## ✅ CI/CD PIPELINE (GitHub Actions)

### Workflow: `mlops-pipeline.yml`

```yaml
name: MLOps Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-test.txt
      
      - name: Lint with flake8
        run: |
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
      
      - name: Test with pytest
        run: |
          pytest tests/ --cov=src/ --cov-report=xml
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker images
        run: |
          docker-compose build
      
      - name: Run integration tests
        run: |
          docker-compose up -d
          sleep 10
          pytest integration_tests/
          docker-compose down

  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        run: |
          echo "Deploying to production..."
          # Example: Deploy to cloud (GCP, AWS, or local VPS)
          ssh -i ${{ secrets.DEPLOY_KEY }} user@prod.server "cd /app && git pull && docker-compose up -d"
```

---

## 📂 PROJECT REPOSITORY STRUCTURE

```
stock-vision-mlops/
│
├── .github/
│   └── workflows/
│       ├── mlops-pipeline.yml
│       └── code-quality.yml
│
├── airflow/
│   ├── dags/
│   │   └── stock_prediction_pipeline.py
│   └── logs/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── features/
│   │   └── .gitkeep
│   └── .dvc
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py  # FastAPI app
│   │   ├── api/
│   │   │   ├── endpoints.py
│   │   │   └── schemas.py
│   │   ├── models/
│   │   │   ├── price_model.py
│   │   │   ├── pattern_model.py
│   │   │   └── ensemble.py
│   │   └── utils/
│   │       ├── db.py
│   │       ├── mlflow_utils.py
│   │       └── preprocessing.py
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_models.py
│   │   └── test_preprocessing.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── chatbot.py  # Gradio interface
│   ├── Dockerfile.gradio
│   └── requirements.txt
│
├── notebooks/
│   ├── 01_eda_vnstock.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_lr.ipynb
│   ├── 04_model_training_lstm.ipynb
│   └── 05_model_evaluation.ipynb
│
├── scripts/
│   ├── fetch_data.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── deploy_models.py
│
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── model_performance.json
│   │   │   └── api_health.json
│   │   └── datasources/
│   │       └── prometheus.yml
│   └── alerts.yml
│
├── config/
│   ├── config.yaml
│   ├── symbols.json
│   └── model_config.yaml
│
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .dvcignore
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── DEPLOYMENT.md
└── LICENSE
```

---

## 🚦 DEPLOYMENT ROADMAP

### Phase 1: Local Development (Week 1)
- [ ] Set up PostgreSQL locally
- [ ] Implement data pipeline (Airflow DAG)
- [ ] Train models (Colab notebook)
- [ ] Build FastAPI server
- [ ] Build Gradio frontend
- [ ] Docker Compose all services

### Phase 2: Testing & Validation (Week 2)
- [ ] Unit tests for preprocessing
- [ ] Integration tests for API
- [ ] Model backtesting on historical data
- [ ] Performance benchmarking
- [ ] Load testing (concurrent users)

### Phase 3: MLOps Setup (Week 3)
- [ ] MLflow server + registry
- [ ] Prometheus + Grafana monitoring
- [ ] GitHub Actions CI/CD
- [ ] Model versioning & auto-promotion
- [ ] Logging & alerting

### Phase 4: Production Deployment (Week 4)
- [ ] Deploy to cloud (GCP, AWS, or VPS)
- [ ] Set up SSL/TLS
- [ ] Scale FastAPI (multiple workers)
- [ ] Database backups
- [ ] Incident response procedures

---

## 📈 SUCCESS METRICS

### Model Metrics
- Price prediction MAE: < 2.5% (monthly)
- Pattern classification accuracy: > 70%
- Sharpe ratio (if backtested): > 0.5

### System Metrics
- API latency p95: < 100ms
- API availability: > 99%
- Model retraining time: < 1 hour weekly

### Business Metrics
- Chatbot users (monthly): 1000+
- Prediction accuracy (real-world): > 65%
- User satisfaction: > 4.0/5.0

---

## 🔮 FUTURE IMPROVEMENTS

1. **Multi-horizon forecasting:** Predict 1M, 3M, 6M ahead
2. **Portfolio optimization:** Suggest portfolio based on predictions
3. **Risk modeling:** VaR, CVaR per stock
4. **NLP:** News sentiment + predictions
5. **Real-time updates:** WebSocket endpoint for live predictions
6. **Mobile app:** React Native or Flutter client
7. **Advanced features:** Order flow imbalance, gamma exposure
8. **Ensemble methods:** Combine with fundamental analysis
9. **Explainability:** SHAP values for feature importance
10. **A/B testing:** Test different feature sets & models

---

## 📞 CONTACT & SUPPORT

**Project Owner:** AI Research Engineer  
**Repository:** github.com/[your-repo]/stock-vision-mlops  
**Demo:** http://localhost:7860 (after docker-compose up)  
**API Docs:** http://localhost:8000/docs  
**Monitoring:** http://localhost:3000 (Grafana)

---

## 📄 LICENSE

MIT License - Feel free to use for educational & research purposes.

---

**Generated:** November 20, 2025  
**Version:** 1.0.0  
**Status:** Ready for Development ✅
