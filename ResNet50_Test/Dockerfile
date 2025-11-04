# Dockerfile for ResNet50_test
# Educational Research Project - NOT for clinical use

# ⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
# It must NOT be used for clinical or medical diagnostic purposes.

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=42
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p archive data/processed results reports docs logs

# Set permissions
RUN chmod +x scripts/check_status.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python scripts/check_status.py || exit 1

# Default command
CMD ["python", "scripts/check_status.py"]

# Labels
LABEL maintainer="Research Team <research@example.com>"
LABEL description="Breast Density Exploration via ResNet-50 Embeddings"
LABEL version="1.0.0"
LABEL disclaimer="⚠️ RESEARCH PURPOSES ONLY - NOT FOR CLINICAL USE"
