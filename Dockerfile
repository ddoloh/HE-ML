# Dockerfile for HE-ML Engine on Ubuntu 24.04 LTS
FROM ubuntu:24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12, pip, and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Working Directory
WORKDIR /app

# Create virtual environment and upgrade pip
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Requirements and Install Dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY . /app/

# Expose FastAPI Port
EXPOSE 8000

# Container Health Check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch FastAPI Web Application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
