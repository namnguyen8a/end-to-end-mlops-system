FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api ./src/api

ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_STAGE=Staging

EXPOSE 8000

CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
