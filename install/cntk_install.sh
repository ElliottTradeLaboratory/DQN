#!/bin/bash
echo Install CNTK
apt-get install -y wget

cd /
mkdir /usr/local/mklml
wget https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz
tar -xzf mklml_lnx_2018.0.1.20171227.tgz -C /usr/local/mklml
wget --no-verbose -O - https://github.com/01org/mkl-dnn/archive/v0.12.tar.gz | tar -xzf - && \
cd mkl-dnn-0.12 && \
ln -s /usr/local external && \
mkdir -p build && \
cd build && \
cmake .. && \
make && \
make install && \
cd ../.. && \
rm -rf mkl-dnn-0.12

wget https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.3.tar.gz

tar -xzvf ./openmpi-1.10.3.tar.gz
cd openmpi-1.10.3
./configure --prefix=/usr/local/mpi
make -j all
make install

echo export PATH=/usr/local/mpi/bin:\$PATH >> /root/.bashrc
echo export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:\$LD_LIBRARY_PATH >> /root/.bashrc

pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.6.0-cp35-cp35m-linux_x86_64.whl
pip3 install https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.5-cp35-cp35m-linux_x86_64.whl
#pip3 install https://cntk.ai/PythonWheel/GPU/cntk-2.3.1-cp35-cp35m-linux_x86_64.whl

git clone https://github.com/keras-team/keras.git -q -b 2.1.5
cd keras
python3 setup.py install
pip3 install h5py
