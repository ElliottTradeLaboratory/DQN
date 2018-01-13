#
#  For PyTorch =0.3.0.post4
#
FROM nvidia/cuda:8.0-cudnn6-devel

RUN apt-get update \
 && apt-get install -y python-numpy python3-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig git python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
RUN pip3 install torchvision
RUN pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-1.4.1-cp35-cp35m-linux_x86_64.whl
RUN pip3 install sk-video
RUN pip3 install opencv-python
RUN pip3 install Pillow
