# The API's dependencies only; the code is bind-mounted from the checkout by compose.yaml. Linux has no Metal, so torch is
# the CPU build: train on the host (or a CUDA node) and use this container for the UI and the API.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-training.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.14.1,<3" \
 && pip install --no-cache-dir -r requirements.txt watchfiles
