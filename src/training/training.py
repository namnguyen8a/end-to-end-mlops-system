import os
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),  # Also print to console
    ],
)

logger = logging.getLogger(__name__)

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("insurance_weekly_training_v2")
logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
logger.info(f"MLflow experiment: insurance_weekly_training")

# Artifact directory for model/scaler outputs
ARTIFACT_DIR = "../artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
logger.info(f"Artifact directory: {ARTIFACT_DIR}")

DATA_PATH = "../data/processed"
TICKERS = ["BIC", "BMI", "BVH", "MIG", "PGI"]
RANDOM_STATE = 42

logger.info("=" * 60)
logger.info("Starting training pipeline")
logger.info(f"Tickers to train: {TICKERS}")
logger.info(f"Random state: {RANDOM_STATE}")
logger.info(f"Data path: {DATA_PATH}")
logger.info(f"Log file: {log_file}")
logger.info("=" * 60)

results = []
models = {}
scalers = {}

for ticker in TICKERS:
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing ticker: {ticker}")
        logger.info(f"{'='*60}")
        
        # Load data
        file_path = os.path.join(DATA_PATH, f"{ticker}_weekly_clean.csv")
        logger.info(f"Loading data from: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            continue
            
        df = pd.read_csv(file_path, parse_dates=["time"])
        logger.info(f"Loaded {len(df)} rows for {ticker}")

        # Feature engineering
        logger.info("Creating features...")
        df = df.sort_values("time").reset_index(drop=True)
        df["close_lag1"] = df["close"].shift(1)
        df["close_lag2"] = df["close"].shift(2)
        df["ma_5"] = df["close"].rolling(window=5).mean()
        df["ma_10"] = df["close"].rolling(window=10).mean()
        df["std_5"] = df["close"].rolling(window=5).std()
        
        initial_count = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped_count = initial_count - len(df)
        logger.info(f"Dropped {dropped_count} rows with NaN, remaining: {len(df)} rows")

        feature_cols = ["close_lag1", "close_lag2", "ma_5", "ma_10", "std_5"]
        X = df[feature_cols].values
        y = df["close"].values
        logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

        # Train/Val/Test split
        logger.info("Splitting data into train/val/test...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, shuffle=False, random_state=RANDOM_STATE,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, shuffle=False, random_state=RANDOM_STATE,
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Normalize with StandardScaler (fit on train only)
        logger.info("Fitting StandardScaler on training data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Scaling completed")

        # Train model
        logger.info("Training LinearRegression model...")
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        logger.info("Model training completed")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Metrics for {ticker}:")
        logger.info(f"  MAE:  {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  R²:   {r2:.6f}")

        results.append({
            "ticker": ticker,
            "n_samples": len(df),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        models[ticker] = model
        scalers[ticker] = scaler

        # --- MLflow Tracking ---
        logger.info("Logging to MLflow...")
        try:
            import joblib
            
            run_name = f"linear_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                # Set tags
                mlflow.set_tag("feature_engineering", "lag+rolling")
                mlflow.set_tag("model_type", "LinearRegression")
                mlflow.set_tag("ticker", ticker)
                
                # Log parameters
                mlflow.log_param("ticker", ticker)
                mlflow.log_param("n_samples", len(df))
                mlflow.log_param("n_train", len(X_train))
                mlflow.log_param("n_val", len(X_val))
                mlflow.log_param("n_test", len(X_test))
                mlflow.log_param("random_state", RANDOM_STATE)
                mlflow.log_param("features", ",".join(feature_cols))
                
                # Log metrics
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("R2", r2)
                
                # Save and log model artifact
                model_path = os.path.join(ARTIFACT_DIR, f"{ticker}_linear.joblib")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path="artifacts")
                mlflow.sklearn.log_model(model, "model")
                logger.info(f"Model saved to: {model_path}")
                
                # Save and log scaler artifact
                scaler_path = os.path.join(ARTIFACT_DIR, f"{ticker}_scaler.joblib")
                joblib.dump(scaler, scaler_path)
                mlflow.log_artifact(scaler_path, artifact_path="artifacts")
                logger.info(f"Scaler saved to: {scaler_path}")
                
            logger.info(f"MLflow logging completed for {ticker} (run: {run_name})")
        except Exception as mlflow_error:
            logger.warning(f"MLflow logging failed for {ticker}: {str(mlflow_error)}")
            logger.warning("Continuing without MLflow logging...")
            
        logger.info(f"✓ Successfully completed training for {ticker}")
        
    except Exception as e:
        logger.error(f"✗ Error processing {ticker}: {str(e)}", exc_info=True)
        continue

results_df = pd.DataFrame(results)
print(results_df)

logger.info("\n" + "=" * 60)
logger.info("Training pipeline completed")
logger.info(f"Successfully trained {len(results)} tickers")
logger.info("=" * 60)
logger.info("\nFinal Results Summary:")
for _, row in results_df.iterrows():
    logger.info(f"  {row['ticker']}: MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, R²={row['R2']:.4f}")
logger.info(f"\nLog file saved to: {log_file}")
