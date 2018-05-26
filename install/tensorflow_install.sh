#!/bin/bash
echo Install Tensorflow
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp35-cp35m-linux_x86_64.whl
git clone https://github.com/keras-team/keras.git -q -b 2.1.2
cd keras
python3 setup.py install
pip install h5py
