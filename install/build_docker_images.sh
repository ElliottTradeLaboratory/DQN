#!/bin/bash


if [ -z "$1" ]; then
    echo [usage] ./build_docker_images.sh [pytorch, tensorflow, cntk, mxnet or all]
    exit 1
fi
frameworks="pytorch cntk tensorflow mxnet"
for framework in $frameworks; do
    if [ "$framework" = "$1" -o "$1" = "all" ]; then
        case "$framework" in
            "mxnet" ) cp Dockerfile_cuda8_cudnn5 Dockerfile;;
            "pytorch" ) cp Dockerfile_cuda8_cudnn6 Dockerfile;;
            "tensorflow" ) cp Dockerfile_cuda8_cudnn6 Dockerfile;;
            "cntk" ) cp Dockerfile_cuda8_cudnn6 Dockerfile;;
        esac
        nvidia-docker build -t $1 --build-arg FRAMEWORK=$1 .
    fi
done
nvidia-docker images
