# Use Python 3.9 as base
FROM python:3.9-slim

# Install system dependencies (Tesseract for OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set work directory
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Pre-pull the model during build (optional, makes it faster on startup)
# Note: Ollama needs to be running to pull. We handle this in the start script.
COPY start.sh .
RUN chmod +x start.sh

# Expose the API port
EXPOSE 5050

# Start command
CMD ["./start.sh"]
