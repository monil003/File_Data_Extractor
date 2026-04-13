#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start (important!)
sleep 5

# Pull the extraction model (this might take a minute on first run)
echo "Ensuring extraction model llama3.2:3b is available..."
ollama pull llama3.2:3b

# Start the Flask app using Gunicorn (production grade)
# Use PORT env variable supplied by Render, default to 5050 if not set
echo "Starting Flask API on port ${PORT:-5050}..."
gunicorn --bind 0.0.0.0:${PORT:-5050} --timeout 120 app:app
