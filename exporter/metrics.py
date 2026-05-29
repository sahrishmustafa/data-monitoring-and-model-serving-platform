# exporter/metrics.py
from prometheus_client import Gauge, Counter, Histogram

# Current model validation accuracy (0.0 to 1.0)
model_accuracy = Gauge(
    "model_accuracy",
    "Current validation accuracy of the deployed model"
)

# Total records ingested since startup
records_processed_total = Counter(
    "records_processed_total",
    "Total number of records ingested from the API"
)

# How many times the model has been retrained
retrain_count_total = Counter(
    "retrain_count_total",
    "Total number of model retraining events"
)

# 1 if drift is currently detected, 0 otherwise
distribution_drift_detected = Gauge(
    "distribution_drift_detected",
    "Set to 1 when distribution drift is detected in the current batch"
)

# Number of features added to the schema since startup
feature_added = Counter(
    "feature_added",
    "Number of features added to the schema since startup"
)

# Number of features removed from the schema since startup
feature_removed = Counter(
    "feature_removed",
    "Number of features removed from the schema since startup"
)

# Number of times the /records API returned 503
datalake_unavailable = Counter(
    "datalake_unavailable",
    "Number of times the data source returned 503"
)

# Latency histogram for /predict calls
response_delay_seconds = Histogram(
    "response_delay_seconds",
    "Latency of each /predict API call in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
