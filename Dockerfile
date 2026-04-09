FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a dedicated non-root user
RUN groupadd --system appgroup && useradd --system --gid appgroup --create-home appuser

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create generated directory for runtime state
RUN mkdir -p generated && chown -R appuser:appgroup generated

# Copy startup helper
COPY scripts/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_DEBUG=False
ENV FLASK_ENV=production

# Run with gunicorn via verified entrypoint
USER appuser
CMD ["/usr/local/bin/start.sh"]
