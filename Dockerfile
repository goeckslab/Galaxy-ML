FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc zlib1g-dev graphviz && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/galaxy-ml
COPY . .
RUN python -m pip install --no-cache-dir . pydot && \
    apt-get purge -y gcc zlib1g-dev && apt-get -y autoremove && apt-get clean
