#!/usr/bin/env bash
# deploy/deploy.sh
# Builds the Docker image, pushes to Docker Hub, SSHes into EC2 and runs it.
# Environment variables required:
#   DOCKER_USERNAME, DOCKER_PASSWORD, EC2_HOST, EC2_USER, EC2_SSH_KEY

set -euo pipefail

IMAGE_NAME="${DOCKER_USERNAME}/mlops-serving"
TAG="latest"

echo "=== Building Docker image ==="
docker build -t "${IMAGE_NAME}:${TAG}" .

echo "=== Logging in to Docker Hub ==="
echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin

echo "=== Pushing image ==="
docker push "${IMAGE_NAME}:${TAG}"

echo "=== Deploying to EC2 ==="
# Write the private key to a temp file
KEY_FILE=$(mktemp)
echo "${EC2_SSH_KEY}" > "${KEY_FILE}"
chmod 600 "${KEY_FILE}"

ssh -i "${KEY_FILE}" -o StrictHostKeyChecking=no \
    "${EC2_USER}@${EC2_HOST}" << REMOTE
    # Pull latest image
    docker pull ${IMAGE_NAME}:${TAG}

    # Stop and remove old container (ignore errors if it doesn't exist)
    docker stop mlops-serving 2>/dev/null || true
    docker rm   mlops-serving 2>/dev/null || true

    # Start new container
    docker run -d \
        --name mlops-serving \
        --restart unless-stopped \
        -p 8000:8000 \
        -e SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL}" \
        ${IMAGE_NAME}:${TAG}

    # Verify health
    sleep 5
    curl -sf http://localhost:8000/health && echo "Health check passed!"
REMOTE

rm -f "${KEY_FILE}"
echo "=== Deployment complete ==="
