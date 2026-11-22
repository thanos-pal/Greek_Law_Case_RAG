# Multi-stage build for Greek Law Case RAG API
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Development stage with test dependencies
FROM base as development

# Switch back to root to install test dependencies
USER root

# Install test dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov pytest-mock httpx

# Switch back to non-root user
USER appuser

# Default command for development (with reload)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Test stage for running unit tests
FROM development as test

# Copy test files explicitly
COPY tests/ ./tests/

# Run tests (command will be overridden by docker compose)
CMD ["python", "-m", "pytest", "tests/test_api.py", "-v", "--tb=short", "--asyncio-mode=auto"]