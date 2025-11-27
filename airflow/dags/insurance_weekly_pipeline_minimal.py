from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/opt/mlops-system"
PROJECT_PYTHON = "/usr/local/bin/python"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="insurance_weekly_pipeline_minimal",
    default_args=default_args,
    start_date=datetime(2025, 11, 1),
    schedule_interval="0 3 * * 1",  # every Monday 03:00
    catchup=False,
    tags=["mlops", "insurance"],
) as dag:

    # 1) Data ingestion
    ingest_raw = BashOperator(
        task_id="ingest_raw_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PROJECT_PYTHON} src/ingest/vnstock_client.py"
        ),
    )

    # 2) Training + evaluation + promotion
    train_eval_promote = BashOperator(
        task_id="train_eval_promote",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PROJECT_PYTHON} src/preprocess/processed_data.py && "
            f"{PROJECT_PYTHON} src/utils/upload_processed_to_gcs.py && "
            f"{PROJECT_PYTHON} src/training/training.py && "
            f"{PROJECT_PYTHON} src/utils/promote_best_models.py"
        ),
    )

    ingest_raw >> train_eval_promote
