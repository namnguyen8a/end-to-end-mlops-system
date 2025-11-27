import os
from mlflow.tracking import MlflowClient


TICKERS = ["BIC", "BMI", "BVH", "MIG", "PGI"]
METRIC_NAME = "R2"


def promote_best_models():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"Using MLflow tracking URI: {tracking_uri}")
    client = MlflowClient(tracking_uri=tracking_uri)

    for ticker in TICKERS:
        model_name = f"{ticker}_weekly_linear"
        print(f"\n=== Processing model: {model_name} ===")

        versions = client.search_model_versions(f"name='{model_name}'")
        staging = [v for v in versions if v.current_stage == "Staging"]
        prod = [v for v in versions if v.current_stage == "Production"]

        if not staging:
            print(f"[{ticker}] No staging versions found; skipping.")
            continue

        # Latest staging version
        new = sorted(staging, key=lambda v: int(v.version))[-1]
        new_run = client.get_run(new.run_id)
        new_metric = new_run.data.metrics.get(METRIC_NAME)

        if new_metric is None:
            print(f"[{ticker}] Staging v{new.version} has no {METRIC_NAME} metric; skipping.")
            continue

        if not prod:
            print(f"[{ticker}] No production version; promoting v{new.version} to Production.")
            client.transition_model_version_stage(
                name=model_name,
                version=new.version,
                stage="Production",
                archive_existing_versions=True,
            )
            continue

        current = prod[0]
        curr_run = client.get_run(current.run_id)
        curr_metric = curr_run.data.metrics.get(METRIC_NAME)

        print(
            f"[{ticker}] Prod v{current.version} {METRIC_NAME}={curr_metric}, "
            f"Staging v{new.version} {METRIC_NAME}={new_metric}"
        )

        # Higher R2 is better
        if curr_metric is None or new_metric > curr_metric:
            print(f"[{ticker}] Promoting v{new.version} to Production.")
            client.transition_model_version_stage(
                name=model_name,
                version=new.version,
                stage="Production",
                archive_existing_versions=True,
            )
        else:
            print(f"[{ticker}] New staging version is not better; keep current Production.")


if __name__ == "__main__":
    promote_best_models()
