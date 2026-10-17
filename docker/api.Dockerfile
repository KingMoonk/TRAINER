# The API's dependencies only; the code is bind-mounted from the checkout by compose.yaml. This container is for an NVIDIA
# GPU (compose.yaml reserves it): torch is the CUDA 12.6 build, as in requirements-gpu.txt. A Mac does not run TRAINER in Docker
# (a Linux VM has no Metal); see docs/dev-compose.md. The API refuses to start here if the GPU did not get through.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-training.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 "torch>=2.14.1,<3" \
 && pip install --no-cache-dir -r requirements.txt watchfiles
