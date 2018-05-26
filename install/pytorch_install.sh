#!/bin/bash
echo Install PyTorch

pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
pip install torchvision
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp35-cp35m-linux_x86_64.whl
