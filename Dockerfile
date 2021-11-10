FROM ubuntu:20.04

ENV VERSION=0.9.1
RUN apt-get update && \
    apt-get install -y git python3.7 python3-pip unzip wget && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN pip install Galaxy-ML==$VERSION pydot

RUN  DEBIAN_FRONTEND="noninteractive" apt-get install -y graphviz