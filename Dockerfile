FROM python:3.10-slim

ENV VERSION=v0.10.0
RUN apt-get update && \
    apt-get install -y git unzip wget && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y graphviz

RUN pip install 'https://github.com/goeckslab/Galaxy-ML.git@${VERSION}' pydot

RUN apt-get -y autoremove && apt-get clean && pip cache clean