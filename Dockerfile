# DocVision - CPU Dockerfile
# =========================
# Production-ready container for document AI processing

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # PDF processing
    poppler-utils \
    # Build essentials for some Python packages
    gcc \
    g++ \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN useradd -m -u 1000 docvision

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for models, output, artifacts, cache
RUN mkdir -p /app/models /app/output /app/artifacts /app/.cache \
    && chown -R docvision:docvision /app

# Switch to non-root user
USER docvision

# Download models during build (optional, can be done at runtime)
# Uncomment to pre-download models:
# RUN python -c "from transformers import AutoProcessor, AutoModel; \
#     AutoProcessor.from_pretrained('microsoft/trocr-base-printed'); \
#     AutoProcessor.from_pretrained('microsoft/trocr-base-handwritten')"

# Expose web UI port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health').raise_for_status()" || exit 1

# Default command - run web server
CMD ["python", "-m", "uvicorn", "docvision.web.app:app", "--host", "0.0.0.0", "--port", "8080"]

# Alternative commands:
# CLI processing: docker run docvision docvision process /data/doc.pdf
# Batch processing: docker run docvision docvision batch /data --output /output
