# Start from a lightweight Python base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies (Prophet needs some)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpython3-dev \
    libatlas-base-dev \
    libffi-dev \
    libbz2-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code into the container
COPY . .

# Set the default command
CMD ["python", "ProphetQLSTMV3.py"]
