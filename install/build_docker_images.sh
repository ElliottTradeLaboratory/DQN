#!/bin/bash


echo "Build Docker image for PyTorch, Tensorflow, CNTK"
cp Dockerfile_cuda8_cudnn6 Dockerfile
nvidia-docker build -t dqn_test --no-cache --build-arg FRAMEWORK"=pytorch tensorflow cntk" .

#echo "Build Docker image for MXNet"
#cp Dockerfile_cuda8_cudnn5 Dockerfile
#nvidia-docker build -t dqn_4_mxnet_test --build-arg FRAMEWORK=mxnet .
