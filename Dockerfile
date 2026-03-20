# Dockerfile — Reproducible environment for brain tumor classification
# Provides a consistent CUDA-enabled environment with all dependencies
# pinned to exact versions for reproducibility.
#
# Usage:
#   docker build -t brain-tumor-v2 .
#   docker run --gpus all -v $(pwd)/data:/app/data brain-tumor-v2 make test
#
# Reference: Section 4 tech stack and Section 12 reproducibility checklist

FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for medical imaging libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  wget \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency specification first for Docker layer caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy project source code
COPY . .

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Default command: run tests to verify environment
CMD ["pytest", "tests/", "-v", "--tb=short"]
