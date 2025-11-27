FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir requests==2.31.0

COPY grafana_setup.py .

CMD ["python", "grafana_setup.py"]
