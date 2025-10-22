FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      jq python3 python3-pip git curl ca-certificates make g++ coreutils docker.io && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY tb_harness.py /app/tb_harness.py

ENTRYPOINT ["python3", "/app/tb_harness.py"]
