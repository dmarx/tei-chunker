# Use slim python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install the package
COPY . .
RUN pip install --no-cache-dir -e .

# Create a non-root user
RUN useradd -m -u 1000 chunker
USER chunker

# Run the service
ENTRYPOINT ["python", "-m", "tei_chunker.service"]
