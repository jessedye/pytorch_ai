# Use a python package for full support 3.12 breaks distutils
FROM python:3.11

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
COPY src/pyproject.toml src/requirements.txt ./

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY src/ .

RUN mkdir /app/img

EXPOSE 8080

# Set the entry point or default command
CMD ["uvicorn", "ai:app", "--host", "0.0.0.0", "--port", "8080"]
