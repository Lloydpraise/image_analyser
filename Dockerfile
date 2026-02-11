FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install CPU-only versions to save space and time
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the code
COPY . .

# Railway uses $PORT variable
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]