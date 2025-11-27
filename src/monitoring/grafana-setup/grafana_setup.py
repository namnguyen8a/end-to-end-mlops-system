import time

import requests

GRAFANA_URL = "http://localhost:3000"
AUTH = ("admin", "admin")  # Default credentials
HEADERS = {"Content-Type": "application/json"}


def wait_for_grafana():
    print("Waiting for Grafana to be ready...")
    for _ in range(30):
        try:
            requests.get(GRAFANA_URL)
            return True
        except Exception as e:
            print(f"Error waiting for Grafana: {e}")
            time.sleep(2)
    return False


def setup_datasource():
    print("Configuring Prometheus Datasource...")
    prometheus_payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://prometheus:9090",
        "access": "proxy",
        "isDefault": True,
    }
    resp = requests.post(
        f"{GRAFANA_URL}/api/datasources",
        auth=AUTH,
        json=prometheus_payload,
        headers=HEADERS,
    )
    print(f"Prometheus Datasource: {resp.status_code}")

    print("Configuring Loki Datasource...")
    loki_payload = {
        "name": "Loki",
        "type": "loki",
        "url": "http://loki:3100",
        "access": "proxy",
        "isDefault": False,
    }
    resp = requests.post(
        f"{GRAFANA_URL}/api/datasources", auth=AUTH, json=loki_payload, headers=HEADERS
    )
    print(f"Loki Datasource: {resp.status_code}")


def setup_dashboard():
    print("Creating Monitoring Dashboard...")
    dashboard_payload = {
        "dashboard": {
            "id": None,
            "title": "ML Service Health",
            "tags": ["mlops"],
            "timezone": "browser",
            "panels": [
                {
                    "title": "Requests per Second (RPS)",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "targets": [
                        {
                            "expr": "rate(http_requests_total[1m])",
                            "legendFormat": "{{handler}}",
                        }
                    ],
                },
                {
                    "title": "99th Percentile Latency",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[1m])) by (le))"
                        }
                    ],
                },
                {
                    "title": "Total Predictions",
                    "type": "stat",
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "targets": [
                        {"expr": "sum(http_requests_total{handler='/predict'})"}
                    ],
                },
                {
                    "title": "ML Service Logs",
                    "type": "logs",
                    "gridPos": {"h": 12, "w": 24, "x": 0, "y": 12},
                    "datasource": {"type": "loki", "uid": "loki"},
                    "targets": [{"expr": '{container="ml_service"}', "refId": "A"}],
                    "options": {
                        "showTime": True,
                        "showLabels": True,
                        "showCommonLabels": False,
                        "wrapLogMessage": True,
                        "sortOrder": "Descending",
                    },
                },
            ],
            "refresh": "5s",
        },
        "overwrite": True,
    }
    resp = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        auth=AUTH,
        json=dashboard_payload,
        headers=HEADERS,
    )
    print(f"Dashboard Status: {resp.status_code} - {resp.text}")


if __name__ == "__main__":
    if wait_for_grafana():
        setup_datasource()
        setup_dashboard()
        print("Grafana setup complete!")
    else:
        print("Could not connect to Grafana.")