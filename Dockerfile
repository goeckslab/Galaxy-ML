FROM python:3.9-slim

ENV VERSION=v0.10.0
RUN apt-get update && \
    apt-get install -y gcc git libz-dev unzip wget && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y graphviz

RUN pip install 'git+https://github.com/goeckslab/Galaxy-ML.git'@$VERSION pydot && \
    pip cache purge

RUN apt-get purge -y gcc libz-dev && apt-get -y autoremove && apt-get clean