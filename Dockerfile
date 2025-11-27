FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/ src/
COPY data/processed/ data/processed/

ENV PYTHONPATH=/app
ENV DATA_PATH=/app/data/processed
# MLFLOW_TRACKING_URI should be set when running the container
# Default works with --network host, override for other setups
ENV MLFLOW_TRACKING_URI=http://localhost:5000

CMD ["python3", "src/api/api.py"]

