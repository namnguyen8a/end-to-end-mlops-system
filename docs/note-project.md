### **Project Goal: Build a Fully Containerized, Production-Grade MLOps System**

**1. Project Overview & Core Components**

*   [ ] Design and implement an end-to-end machine learning system using MLflow, Airflow, Docker, FastAPI, Grafana, and Pytest.
*   [ ] Ensure the project is reproducible from scratch with a few commands.

**A. Experiment Tracking with MLflow**
*   [ ] Integrate MLflow into the training pipeline.
*   [ ] Log model parameters during training.
*   [ ] Log model metrics (e.g., accuracy, F1-score).
*   [ ] Log model artifacts (e.g., the trained model file).
*   [ ] Register the best-performing model in the MLflow Model Registry.
*   [ ] Manage model version promotion (e.g., from Staging to Production).

**B. Model Serving with FastAPI**
*   [ ] Create a REST API service using FastAPI.
*   [ ] Load the latest "Production" stage model from the MLflow Model Registry.
*   [ ] Implement a `/predict` endpoint for model inference.
*   [ ] Implement a `/health` endpoint to check service status.
*   [ ] Use Pydantic to validate input data for the `/predict` endpoint.
*   [ ] Ensure the API returns results in JSON format.
*   [ ] Create Pytest-based tests for the API endpoints.

**C. Pipeline Orchestration with Apache Airflow**
*   [ ] Design an Airflow Directed Acyclic Graph (DAG).
*   [ ] Create a task for data ingestion.
*   [ ] Create a task for model training.
*   [ ] Create a task for model evaluation.
*   [ ] Create a task for promoting the model in MLflow if it meets performance criteria.
*   [ ] Schedule the DAG for automatic execution.

**D. Containerization with Docker & Docker Compose**
*   [ ] Create a Dockerfile for the MLflow service.
*   [ ] Create a Dockerfile for the FastAPI service.
*   [ ] Create a Dockerfile/setup for the Airflow service.
*   [ ] Create a Dockerfile/setup for the Grafana service.
*   [ ] Create a `docker-compose.yml` file to orchestrate all services.
*   [ ] Ensure the entire system can be launched with a single `docker-compose up` command.

**E. Monitoring with Grafana**
*   [ ] Expose metrics from the FastAPI service in a Prometheus-compatible format.
*   [ ] Integrate Grafana into the Docker Compose setup.
*   [ ] Create a Grafana dashboard to visualize key system metrics.
*   [ ] Monitor CPU and memory usage of services.
*   [ ] Monitor API latency.
*   [ ] Monitor total request count to the API.

**F. Testing with Pytest**
*   [ ] Implement automated tests for the FastAPI service.
*   [ ] Implement automated tests for the Airflow pipeline logic (e.g., DAG integrity).
*   [ ] Implement unit tests for any helper functions or data processing logic.
*   [ ] Configure the project to run the entire test suite with a single command.

---

**2. Bonus Challenges**

*   [ ] Implement a CI/CD pipeline with GitHub Actions to automatically build and test containers on code push.
*   [ ] Use DVC (Data Version Control) to version the dataset.
*   [ ] Deploy the containerized system to a cloud service (e.g., Google Cloud Run, AWS ECS, or Azure Container Apps).
*   [ ] Implement a mechanism for automated model retraining triggered by performance degradation or data drift metrics.

---

**3. Submission Requirements**

*   [ ] Create a public GitHub repository for the project.
*   [ ] Include all source code.
*   [ ] Include all Docker configurations (`Dockerfile`, `docker-compose.yml`).
*   [ ] Write a comprehensive `README.md` file.
    *   [ ] Include clear setup and installation instructions.
    *   [ ] Provide instructions on how to run the system.
    *   [ ] Include a brief overview of the system architecture.
*   [ ] Include screenshots in the README or repository showing:
    *   [ ] The MLflow UI with logged experiments.
    *   [ ] The Airflow DAG running successfully.
    *   [ ] The FastAPI `/docs` page.
    *   [ ] The Grafana monitoring dashboard.
    *   [ ] The output of the Pytest test suite.
*   [ ] (Optional) Include a short demo video.

---

**4. Evaluation Criteria Checklist**

*   **[ ] Functionality (20 points):** Do all services work together seamlessly?
*   **[ ] Code Quality (10 points):** Is the code modular, clean, and well-documented?
*   **[ ] MLflow Integration (10 points):** Is tracking of runs, metrics, and model versioning implemented correctly?
*   **[ ] Airflow Pipeline (15 points):** Is the DAG functional with correct task sequencing and automation?
*   **[ ] Model Serving API (10 points):** Is the FastAPI service reliable with working, validated endpoints?
*   **[ ] Containerization (10 points):** Is Docker and Docker Compose used correctly?
*   **[ ] Monitoring (10 points):** Is the Grafana dashboard functional and displaying relevant metrics?
*   **[ ] Testing (Pytest) (10 points):** Is there a comprehensive test suite integrated into the workflow?
*   **[ ] Documentation (5 points):** Is the README clear and comprehensive?