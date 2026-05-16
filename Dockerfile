# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements from src and install
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from src into /app
COPY src/ .

# Expose the port (Railway will provide this via environment variable)
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Command to run the application
CMD ["python", "app.py"]
