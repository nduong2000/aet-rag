# rag_app/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
# Standard port for Cloud Run
ENV PORT 8080

# Create and set the working directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    

# Copy the requirements file and install Python dependencies
COPY requirements-main.txt .
RUN pip install --no-cache-dir -r requirements-main.txt

# Copy the rest of the application code into the image
COPY ./documents ./documents
COPY ./chroma_db_data ./chroma_db_data
COPY ./templates ./templates
COPY field_definitions.json .
COPY main.py .
#COPY .env .  # If you have any environment variables

# Make sure chroma_db_data directory exists
RUN mkdir -p ./chroma_db_data

# Create a healthcheck file for startup checks
RUN echo "OK" > healthcheck.txt

# Expose the port the app runs on
EXPOSE $PORT

# Define the command to run the application using Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app