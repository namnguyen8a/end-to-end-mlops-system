FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==3.5.1 pymysql==1.1.1 google-cloud-storage==3.6.0

# GCP creds will be mounted at runtime
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "mysql+pymysql://mlflowuser:lp6tQtvD%40I%5DS%24J%26o@34.46.193.172:3306/mlflowdb", \
     "--default-artifact-root", "gs://mlops-tiker-bucket/mlflow-artifacts", \
     "--allowed-hosts", "*", \
     "--cors-allowed-origins", "*"]
