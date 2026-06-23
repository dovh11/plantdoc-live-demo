# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Builder
#   Install all Python dependencies into an isolated prefix so the final image
#   never needs build tools (gcc, g++, etc.).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Prevents .pyc files and enables unbuffered stdout/stderr in the builder too
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /install

# Install only the build-time system libraries needed by opencv-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first → Docker layer cache: deps rebuild only when
# requirements.txt changes, not on every source-file edit.
COPY requirements.txt .

# Install into a separate prefix so we can COPY it cleanly into the final image
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install/pkgs -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Runtime
#   Lean image: no build tools, no pip, minimal system packages.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="plantdoc-live-demo" \
      description="Plant Disease Detection – FastAPI + YOLOv8 ONNX" \
      version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Make the installed packages visible to Python
    PYTHONPATH=/app/pkgs/lib/python3.11/site-packages

WORKDIR /app

# Runtime-only system libs required by opencv-headless & onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install/pkgs /app/pkgs

# Copy application source
COPY assets/   ./assets/
COPY main.py   .

# Create the static directory; Phase 3 will populate it
RUN mkdir -p static

# Copy static files if they already exist (no-op when the dir is empty)
COPY static/   ./static/

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the application port
EXPOSE 8000

# Health-check so orchestrators know when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start uvicorn with production-grade settings
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--log-level", "info"]
