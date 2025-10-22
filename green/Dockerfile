FROM python:3.11-slim

WORKDIR /app

# Build tools for white agent
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Expect build context at repo root; use -f green/Dockerfile when building
COPY green/run_green.py /app/green/run_green.py
COPY white /app/white

# Build the white agent binary
RUN make -C /app/white

# Default: run the green agent. Mount your dataset at /data when running.
ENTRYPOINT ["python3", "/app/green/run_green.py"]

