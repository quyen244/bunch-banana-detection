# ==============================================================================
# Dockerfile — Banana Bunch Detection API
# Base: python:3.11-slim (CPU-only inference)
# Port: 8000
# Model: mounted at runtime via -v ./models:/app/models
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for OpenCV and general networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (avoids pulling the large CUDA build)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source only (model is volume-mounted at runtime)
COPY backend/ ./backend/

# Placeholder directory — actual model files are mounted via -v
RUN mkdir -p /app/models

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
