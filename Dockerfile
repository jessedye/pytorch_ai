# Use a Debian-based image for Python (slim version for smaller size)
FROM python:3.12.5-slim

# Create app directory
WORKDIR /app

# Update package list and upgrade installed packages
RUN apt-get update && apt-get upgrade -y

# Install system dependencies required for building libraries
RUN apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libssl-dev \
    libffi-dev \
    cmake \
    gfortran \
    liblapack-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY src/requirements.txt ./

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY src/ .




# Install PyTorch with CUDA separately
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
#CPU only
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu


RUN pip install -r requirements.txt
# copy app source
COPY src /app

EXPOSE 8080

RUN ls -lsa

# Set the entry point or default command
CMD ["uvicorn", "ai:app", "--host", "0.0.0.0", "--port", "8080"]
