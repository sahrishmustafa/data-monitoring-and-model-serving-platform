# MLOps End-to-End Pipeline

**Course:** Machine Learning Operations (MLOps) — Spring 2026, FAST NUCES
**Department:** Computer Science / Data Science
**Team Member:** Sahrish Mustafa — 22i0977

---

## Project Description

A production-grade MLOps pipeline that ingests live, evolving data from an HTTP API, trains and automatically retrains a machine learning classifier, deploys it as a REST API on AWS EC2, and monitors the entire system with Prometheus, Grafana, and Alertmanager — with real-time Slack notifications and a fully automated CI/CD pipeline via GitHub Actions.

The system is designed to be self-healing: it detects data drift, schema changes, and model degradation automatically, retrains without manual intervention, and notifies stakeholders through Slack.

---

## System Architecture

```
Data API (port 6500)
        │
        ▼
Ingestion Script (local)
  ├── Schema monitoring
  ├── Drift detection
  └── Triggers retraining
        │
        ▼
Model Training (scikit-learn Random Forest)
  └── Saves versioned model_vN.pkl
        │
        ▼
FastAPI Server (Docker on AWS EC2 — port 8000)
  ├── POST /predict
  ├── GET  /metrics
  └── GET  /health
        │
        ▼
Prometheus (local Docker Compose — scrapes EC2 every 15s)
  ├── Evaluates 7 alert rules
  └── Sends firing alerts to Alertmanager
        │
        ├──▶ Grafana (dashboards)
        └──▶ Alertmanager ──▶ Slack (#mlops-alerts)

GitHub Actions (on every push to main)
  └── Lint → Test → Build Docker Image → Deploy to EC2
```

---

## EC2 Public IP

```
100.54.209.53
```

The FastAPI inference server is publicly accessible on port 8000.

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/course-project-sahrishmustafa.git
cd course-project-sahrishmustafa
```

### 2. Create and activate Python virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
nano .env   # or open in any text editor
```

Fill in the following values in `.env`:

```
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
DRIFT_THRESHOLD=0.2
RETRAIN_ACCURACY_THRESHOLD=0.80
DATA_API_URL=http://149.40.228.124:6500/records
POLL_INTERVAL_SECONDS=30
```

> Never commit your `.env` file. It is listed in `.gitignore`.

### 5. Train the initial model

```bash
python model/train.py
```

This reads from `data/records.csv`, trains a Random Forest classifier, enforces validation accuracy ≥ 0.80, and saves the model as `model/model_v1.pkl`.

### 6. Run the ingestion script

```bash
python ingestion/ingestion.py
```

This polls the data API every 30 seconds, saves records locally, monitors for schema changes and distribution drift, and triggers retraining automatically when needed.

### 7. Start the observability stack

```bash
docker compose up -d
```

This starts three containers:
- **Prometheus** at `http://localhost:9090`
- **Grafana** at `http://localhost:3000` (login: admin / admin)
- **Alertmanager** at `http://localhost:9093`

To verify Prometheus is scraping correctly, go to `http://localhost:9090/targets` — the EC2 target should show as **UP**.

### 8. Configure Grafana

1. Open `http://localhost:3000`
2. Go to Connections → Data sources → Add new → Prometheus
3. Set URL to `http://prometheus:9090`
4. Click Save & Test
5. Go to Dashboards → Import → upload `grafana/dashboards/mlops_dashboard.json`

---

## AWS EC2 — Testing the Deployed Endpoints

The model server is running inside Docker on EC2. Test it directly:

### Health check
```bash
curl http://100.54.209.53:8000/health
# Expected: {"status":"ok"}
```

### Run a prediction
```bash
curl -X POST http://100.54.209.53:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'
# Expected: {"prediction": 1, "confidence": 0.87}
```

### View raw Prometheus metrics
```bash
curl http://100.54.209.53:8000/metrics
```

This returns all 8 metrics in Prometheus text format, which Prometheus scrapes automatically every 15 seconds.

---

## Slack Webhook Configuration

### Setting up the webhook

1. Go to `https://api.slack.com/apps` → Create New App → From Scratch
2. Name it `MLOps Alerts`, select your workspace
3. Left panel → Incoming Webhooks → Activate Incoming Webhooks
4. Click Add New Webhook to Workspace → select channel `#mlops-alerts`
5. Copy the webhook URL

### Configuring it locally

Add the URL to your `.env` file (wrap in double quotes to handle special characters):

```
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

Then restart the observability stack:

```bash
docker compose down && docker compose up -d
```

### Configuring it for CI/CD

Add it as a GitHub Actions secret:
- Go to your repo → Settings → Secrets and Variables → Actions
- Add secret named `SLACK_WEBHOOK_URL` with your webhook URL

Alertmanager routes all 7 alerts to the `#mlops-alerts` channel automatically. Resolved alerts also send a notification when conditions return to normal.

---

## Alert Descriptions and How to Trigger Them

All 7 alerts are defined in `prometheus/alert_rules.yml` and routed through Alertmanager to Slack.

| # | Alert Name | Condition | Slack Message | How to Trigger |
|---|---|---|---|---|
| 1 | DataLakeUnavailable | `increase(datalake_unavailable_total[1m]) > 0` | Data source returned 503. Check API availability. | API endpoint returns 503 — handled automatically by ingestion script |
| 2 | FeatureAdded | `increase(feature_added_total[1m]) > 0` | New feature detected in schema. Retraining may be required. | New column appears in API response schema |
| 3 | FeatureRemoved | `increase(feature_removed_total[1m]) > 0` | Feature dropped from schema. Verify pipeline compatibility. | Existing column disappears from API response schema |
| 4 | DistributionDrift | `distribution_drift_detected == 1` | Data distribution drift detected. Model may be stale. | Feature mean shifts beyond configured threshold |
| 5 | FeatureDriftDetected | `distribution_drift_detected > 0` | Feature-level drift flagged. Investigate upstream data. | Same as above — any positive drift value |
| 6 | HighResponseLatency | `histogram_quantile(0.95, rate(response_delay_seconds_bucket[5m])) > 1.0` | P95 response latency exceeded 1 second. | Inference calls taking longer than 1 second |
| 7 | LowModelAccuracy | `model_accuracy < 0.8` | Model accuracy dropped below threshold. Auto-retraining triggered. | Model validation accuracy drops below 80% |

### Triggering all alerts at once (for testing)

A test endpoint is available on the running server:

```bash
curl -X POST http://100.54.209.53:8000/test/trigger-alerts
```

This forces all metric values into alert territory simultaneously. Watch `http://localhost:9090/alerts` — rules will turn red within 30 seconds, and Slack messages will arrive within 60 seconds.

### Resetting back to normal after testing

```bash
curl -X POST http://100.54.209.53:8000/test/reset-alerts
```

Or for a complete reset of all metrics including counters:

```bash
ssh -i mlops-key.pem ubuntu@100.54.209.53
docker restart mlops-serving
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/mlops-ci.yml`) runs automatically on every push to `main`:

| Job | What it does |
|---|---|
| Lint & Test | Runs flake8 linting and pytest unit tests |
| Build & Push | Builds Docker image, pushes to Docker Hub with `:latest` and `:<git-sha>` tags |
| Deploy to EC2 | SSHes into EC2, pulls new image, restarts container, verifies `/health` returns 200 |

### Required GitHub Secrets

Configure these in Settings → Secrets and Variables → Actions:

| Secret | Value |
|---|---|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub access token |
| `EC2_SSH_KEY` | Full contents of `.pem` key file including header and footer |
| `EC2_HOST` | `100.54.209.53` |
| `EC2_USER` | `ubuntu` |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook URL |

---

## Running the Full System Locally

```bash
# Terminal 1 — start observability stack
docker compose up -d

# Terminal 2 — start local model server
docker run -d --name mlops-serving -p 8000:8000 mlops-serving

# Terminal 3 — start ingestion loop
python ingestion/ingestion.py

# Terminal 4 — manually trigger retraining if needed
python model/retrain_trigger.py --reason manual
```

---

## Project Structure

```
mlops-project/
├── .github/workflows/mlops-ci.yml     # GitHub Actions CI/CD
├── ingestion/
│   ├── ingestion.py                   # Data fetching & schema monitoring
│   └── drift_detector.py              # Distribution drift logic
├── model/
│   ├── train.py                       # Training script
│   ├── retrain_trigger.py             # Auto-retraining orchestration
│   └── model_v1.pkl                   # Serialized model
├── serving/
│   └── app.py                         # FastAPI inference server
├── exporter/
│   └── metrics.py                     # Prometheus metric definitions
├── prometheus/
│   ├── prometheus.yml                 # Prometheus config
│   └── alert_rules.yml               # 7 alerting rules
├── alertmanager/
│   └── alertmanager.yml              # Slack routing config
├── grafana/dashboards/
│   └── mlops_dashboard.json          # Exported Grafana dashboard
├── deploy/
│   └── deploy.sh                     # AWS deployment script
├── tests/
│   ├── test_schema.py                # Unit test: schema change detection
│   ├── test_drift.py                 # Unit test: drift detection
│   └── test_predict.py              # Unit test: /predict endpoint
├── Dockerfile                        # Container for ML service
├── docker-compose.yml               # Observability stack
├── requirements.txt                 # Python dependencies
├── .env.example                     # Example env vars (no real secrets)
└── README.md                        # This file
```

---

## Video Demo

[[Google Drive Link](https://drive.google.com/drive/folders/1Gyg4tzJ__S5BjLP8apCjNSjOHdLqql7m?usp=sharing)]

The demo covers: live data ingestion, schema and drift monitoring, alert firing in Slack, Grafana dashboard with live data, and a passing GitHub Actions CI/CD run.
