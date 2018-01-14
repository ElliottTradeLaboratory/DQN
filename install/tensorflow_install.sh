#!/bin/bash
echo Install Tensorflow
pip3 install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp35-cp35m-linux_x86_64.whl
git clone https://github.com/keras-team/keras.git -q
cd keras
git checkout -b 2.1.2 -q
python3 setup.py install
pip3 install h5py
