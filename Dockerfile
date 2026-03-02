# Use an official NVIDIA CUDA runtime image based on Ubuntu 24.04
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3.12-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Create a venv and add to PATH
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt /app/

# Install the Python dependencies
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt

# Copy project code into container
COPY . /app/

# Set default command
CMD ["bash"]