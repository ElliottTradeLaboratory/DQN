#!/bin/bash
apt-get update
apt-get install -y python-numpy python3-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig python3-pip
pip3 install --upgrade pip
pip3 install --upgrade setuptools
