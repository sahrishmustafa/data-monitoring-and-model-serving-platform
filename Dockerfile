# Dockerfile (project root)
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY serving/ ./serving/
COPY exporter/ ./exporter/
COPY model/ ./model/
COPY .env.example .env

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
