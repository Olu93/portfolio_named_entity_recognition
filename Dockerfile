# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock README.md ./ 

# Configure Poetry to not create a virtual environment (since we're in a container)
RUN poetry config virtualenvs.create false

# Install dependencies (excluding GPU dependencies for Alpine compatibility)
RUN poetry install --without gpu --no-root

# Copy the source code
COPY src/ ./src/
COPY files/ ./files/
# COPY models/ ./models/

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["python", "src/main.py"] 