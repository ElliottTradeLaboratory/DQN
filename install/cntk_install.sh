#!/bin/bash
echo Install CNTK
pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.6.0-cp35-cp35m-linux_x86_64.whl
pip3 install https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.5-cp35-cp35m-linux_x86_64.whl
#pip3 install https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp35-cp35m-linux_x86_64.whl
ln -s /usr/lib/openmpi/lib/libmpi.so /usr/lib/openmpi/lib/libmpi.so.12
echo # CNTK >> ~/.bashrc
echo export LD_LIBRARY_PATH=/usr/lib/openmpi/lib:$LD_LIBRARY_PATH >> ~/.bashrc
git clone https://github.com/keras-team/keras.git -q -b 2.1.5
cd keras
python3 setup.py install
pip3 install h5py
