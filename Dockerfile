# Base image with CUDA support (adjust for your PyTorch version and CUDA version)
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables for non-interactive apt-get and CUDA paths
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    build-essential \
    libopenblas-dev liblapack-dev libjpeg-dev zlib1g-dev \
    git wget curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA directly from PyTorch's custom index
RUN python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Set working directory
WORKDIR /app

# Copy requirements.txt to leverage Docker caching
COPY ./backend/requirements.txt /app/requirements.txt

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Install all requirements
RUN python3.11 -m pip install -r /app/requirements.txt


# Copy the rest of the application code
COPY ./backend /app

# Expose the API port
EXPOSE 8000

# Set the command to run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# # Stage 1: Build Stage
# FROM nvidia/cuda:12.4.0-base-ubuntu22.04 AS builder

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3.11 python3-pip build-essential \
#     libopenblas-dev liblapack-dev libjpeg-dev zlib1g-dev \
#     git wget curl && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# # Upgrade pip
# RUN python3.11 -m pip install --upgrade pip

# # Install PyTorch with CUDA
# RUN python3.11 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# # Set working directory
# WORKDIR /app

# # Copy requirements.txt
# COPY ./backend/requirements.txt /app/requirements.txt

# # Install Python dependencies
# RUN python3.11 -m pip install --no-cache-dir -r /app/requirements.txt


# # Copy the application code
# COPY ./backend /app

# # Stage 2: Runtime Stage
# FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# # Set working directory
# WORKDIR /app

# # Copy application and dependencies from the builder
# COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
# COPY --from=builder /usr/local/bin /usr/local/bin
# COPY --from=builder /app /app

# # Expose the API port
# EXPOSE 8000

# # Command to run the app
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
