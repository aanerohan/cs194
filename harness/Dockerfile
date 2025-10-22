FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      jq python3 python3-pip git curl ca-certificates make g++ coreutils docker.io && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy

WORKDIR /app
COPY tb_harness.py /app/tb_harness.py

ENTRYPOINT ["python3", "/app/tb_harness.py"]
