#
#  Docker image with cuda8.0 & cudnn5
#
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ARG FRAMEWORK
RUN apt-get update
RUN apt-get install -y  git
RUN git clone --recursive https://github.com/ElliottTradeLaboratory/DQN.git
RUN cd /DQN/install \
  && chmod 777 *.sh \
  && ./dependencies.sh \
  && ./setup.sh $FRAMEWORK
