# DQN implementation with DQN3.0-level performance using Python with PyTorch, MXNet, Tensorflow and CNTK (last two on Keras)

This repository contains an implementation of Deep Q-Network ([Mnih et al., 2015](https://www.nature.com/articles/nature14236)) (DQN) using major deep learning frameworks such as PyTorch, MXNet, Tensorflow and CNTK (last two on Keras) .
These are accuracy reproduced [DQN3.0](https://github.com/deepmind/dqn).

See ["The Reproduction Method of Deep Q-network with DQN3.0-level performance"](https://elliotttradelaboratory.github.io/DQN/) for more details.

## Installation Overview

* The installation requires Linux.<br>
* Strongly recommended use GPU.　Because it takes about 2~3 days to learn 5 million steps even when running on the GTX1080ti.

## Installation instructions

### 1. Run on Docker

#### 1-1. Install nvidia-docker

Install nvidia-docker as follows:
[https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))

#### 1-2. Clone repository

```
cd <clone root dir>
git clone https://github.com/ElliottTradeLaboratory/DQN.git
```

NOTE: If you want see other source code such as alewrap_py and xitari, you can specify `--recursive` option and download them at same time.

#### 1-3. Build Docker images

Build Docker images for any framework as follows:
```
$ cd <clone root dir>/DQN/install
$ ./build_docker_images.sh
　　・
　　・
　　・
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
dqn                 latest              xxxxxxxxxxxx        xxxxxxx             xxxxxx
dqn_mxnet           latest              xxxxxxxxxxxx        xxxxxxx             xxxxxx
$
```
`dqn` is contained PyTorch, Tensorflow-gpu, CNTK, Keras.<br>
`dqn_mxnet` is contained MXNet and Tensorflow(CPU)

#### 1-4. Run DQN
`-v` option for logdir is strongly recommended for training time.

```
$ nvidia-docker run -it --rm dqn [-v <any Host logdir>:<any container logdir>]
root@xxxxxx:/# cd DQM
root@xxxxxx:/DQN# ./run <backend name> <game name> [--logdir <any container logdir>] [options]
```

#### 1-5. Visualization

if use `-v` for logdir with Host logdir `/tmp`
```
$ cd /tmp/<game name>
$ tensorboard --logdir .
```

### 2. Run on your environment without Docker

#### 2-1. Install CUDA and cuDNN

If you want to run with GPU, you need to install CUDA and cuDNN according to the following site:

CUDA  : Download from: https://developer.nvidia.com/cuda-toolkit-archive<br>
　　　　Install guid: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html<br>
cuDNN : Download from: https://developer.nvidia.com/rdp/cudnn-download<br>
　　　　Install guid: [cuDNN5.1](http://developer2.download.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/doc/cudnn_install.txt?4Y7u0FqHrotFcmVuCKOpM2anE-n8iMSBbn9WCrSMFTUFQzXCSGfEIkdPvFi0yoyTYBTKJzIiKiVwvgSYDqnfDzpew8WT1PdIAnXOeStXoMX2meBxzvBWZmNaVc3dt5u8Cv96mWCoTVp87ppWFM22UG1vqwAgwu4pR-W7m7fuHGOfIMYr), [cuDNN6.0](http://developer2.download.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/Doc/cudnn_install-2.txt?5e1fCcgO0eYlHY7zwZH-LBiJJBZRX4pF_wv1Gf3hq1lpsF6Q0pvkc0BkdZKVwfxaT-m8iAjLn0ZV6NRh_-jGp8GCMDnmUmCHtxQ82UQnwQVlrzZebTFGRm5q90Ic8S7UC2SMG0Z-NXlwLQfqOpr7l6YErWhJB1Ai2dc4ggsXjPFAtEx_)

NOTE: The version of CUDA and cuDNN you should install depends on the version of the framework you need.　See [Description](#description) more detail.

#### 2-2. Clone repository

```
$ cd <clone root dir>
$ git clone --recursive https://github.com/ElliottTradeLaboratory/DQN.git
```

#### 2-3. Run Install Script

```
$ cd <clone root dir>/DQN/install
$ ./install_your_linux.sh
```

#### 2-4. Run DQN

```
$ cd <clone root dir>/DQN
$ ./run <backend name> <game name> [options]
```

### 3. Visualization

```
$ cd <log dir>/<game name>
$ tensorboard --logdir .
```

### Description
* _\<clone root dir\>_ is an any directory for cloning repositories from github.<br>
* _\<log dir\>_ is a any directory to output logs for Tensorboard. As default `/tmp`<br>
* _\<framework name\>_ as follows:

_\<framework name\>_ | Install frameworks| CUDA | cuDNN
---------------|-----|-----|-----
`pytorch`[<sup>[2]</sup>](#pytorch_cuda) | PyTorch 0.3.0.post4<br> Tensorflow 1.4.1(cpu)[<sup>[3]</sup>](#tensorflow) | 8.0 | 6.0 
`tensorflow` | Tensorflow-gpu 1.4.1<br>Keras 2.1.2 | 8.0 | 6.0
`cntk` | CNTK 2.3.1<br> Tensorflow 1.4.1(cpu)<br>Keras 2.1.2 | 8.0 | 6.0
`mxnet` | MXNet 1.0.0<br> Tensorflow 1.2.1(cpu) | 8.0 | 5.1

* _\<game name\>_ is:<br>
　For alewrap_py,  a module name of the game exclude `.bin`(e.g. `breakout`).<br>
　For Gym, a env name of Gym(e.g. `Breakout-v0`).<br>
* _\<backend name\>_ as follows:

_\<backend name\>_ | Framework used for networks
---------------|----------
`pytorch` | PyTorch with `torch.nn package`
`pytorch_legacy` | PyTorch with `torch.legacy.nn package`
`tensorflow` | Tensorflow with Keras
`cntk` | CNTK with Keras
`mxnet` | MXNet
