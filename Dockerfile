# Zenith training/generation image.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e ".[cli,tracking,serving]"

EXPOSE 8000

# Train on the bundled corpus:
#   docker run --rm -v "$(pwd)":/app zenith python -m zenith.cli.train
# Serve a checkpoint (POST /generate, /generate/stream):
#   docker run --rm -p 8000:8000 -v "$(pwd)":/app zenith zenith serve -m zenith-lm.pt
CMD ["zenith", "info"]
