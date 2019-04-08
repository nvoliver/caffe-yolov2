FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install git python3-pip caffe-cpu=1.0.0-6 -y --no-install-recommends
RUN pip3 install setuptools 'argparse==1.4.0'