# MLOps End-to-End Pipeline

**Course:** Machine Learning Operations (MLOps) — Spring 2026, FAST NUCES
**Team:** [Your Name] — [Your Roll Number]

## Project Description
An end-to-end production MLOps pipeline that ingests live data, trains and auto-retrains an ML model, deploys it to AWS EC2, and monitors it with Prometheus/Grafana/Alertmanager with Slack alerting and CI/CD via GitHub Actions.

## Architecture
- **Data Ingestion:** Python script polls `http://149.40.228.124:6500/records` every 30s
- **ML Model:** Random Forest classifier (scikit-learn), auto-retrained on drift/schema change
- **Deployment:** FastAPI server in Docker on AWS EC2 (t2.micro)
- **Observability:** Prometheus + Grafana + Alertmanager via Docker Compose
- **Alerting:** 7 Prometheus alert rules routed to Slack
- **CI/CD:** GitHub Actions — lint, test, build Docker image, deploy to EC2

## EC2 Public IP
`YOUR_EC2_IP`

## Setup — Local

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/mlops-project.git
cd mlops-project
```

### 2. Python environment
```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your SLACK_WEBHOOK_URL
```

### 4. Run ingestion
```bash
python ingestion/ingestion.py
```

### 5. Train initial model
```bash
python model/train.py
```

### 6. Start observability stack
```bash
docker compose up -d
```

## Setup — AWS

### Test the deployed endpoints
```bash
curl http://YOUR_EC2_IP:8000/health
curl http://YOUR_EC2_IP:8000/metrics
curl -X POST http://YOUR_EC2_IP:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.0, 3.0]}'
```

## Slack Webhook Configuration
Set SLACK_WEBHOOK_URL in .env (local) or as a GitHub Actions secret (CI/CD).
Alerts go to the #mlops-alerts channel.

## Alert Descriptions and Triggers

| Alert | Trigger |
|---|---|
| DataLakeUnavailable | API returns 503 |
| FeatureAdded | New column in schema |
| FeatureRemoved | Column dropped from schema |
| DistributionDrift | drift_detected gauge == 1 |
| FeatureDriftDetected | drift_detected gauge > 0 |
| HighResponseLatency | P95 latency > 1s |
| LowModelAccuracy | model_accuracy < 0.8 |
