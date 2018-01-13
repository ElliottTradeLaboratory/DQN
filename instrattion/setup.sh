#!/bin/bash

sudo apt-get update
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig git python3-pip
pip3 install --upgrade pip
git clone https://github.com/openai/gym.git
cd gym
pip3 install -e .[atari]
cd /DQN/alewrap_py
python3 setup.py build
python3 setup.py install

