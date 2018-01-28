#
#  Build DQN image with cuda9.0 & cudnn7
#
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ARG FRAMEWORK
RUN apt-get update
RUN apt-get install -y  git
RUN git clone --recursive https://github.com/ElliottTradeLaboratory/DQN.git
RUN cd /DQN/install \
  && ./dependencies.sh \
  && ./setup.sh $FRAMEWORK
