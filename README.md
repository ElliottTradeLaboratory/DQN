# DQN implementation with DQN3.0-level performance through Python using PyTorch, MXNet, Tensorflow and CNTK　[(日本語)](README_jp.md)

This repository including Deep Q-Network[(Mnih et al., 2015)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (DQN)implementations using currentry major Deep Learning Frameworks such as PyTorch, MXNet, Tensorflow and CNTK(both on Keras).<br>
Each framework is used for the implementation of the network part and the optimization part, and learning on the common DQN agent and training platform makes it possible to compare the performance of each framework under the same condition.<br>
These implementation methods and performance are exactly the same as [DQN3.0](https://github.com/deepmind/dqn) which is the reference implementation by [Mnih et al., [2015]](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

See [wiki](https://github.com/ElliottTradeLaboratory/DQN/wiki) for more details.

## Installation Overview

* The installation requires Linux.<br>
* Strongly recommended use GPU.　Because it takes about 7 days to learn 5 million steps even when running on the GTX1080ti using the fastest version with the setting closest to DQN 3.0.

### ◇Deep Learning Frameworks
This implementation uses multiple frameworks, but you can install and run all of them, or you can install and run only one framework.
However, only Tensorflow is necessary for logging.

The easiest way to run without breaking your Linux environment is to use [Docker](https://www.docker.com/) to build those environments from [Dockerfiles](https://github.com/ElliottTradeLaboratory/DQN/tree/master/install) onto Docker as follows "[1. Run on Docker](#1-run-on-docker)".

Particularly, in the case of CNTK, it is useful to use Docker to running parallel train when you needs running parallel with different arguments.　Because, parallelization in CNTK is implemented with MPI, therefore you need to use `mpiexec` and execute like `mpiexec --npernode $num_workers python training.py`[<sup>[1]</sip>](#cntk_mpi).

If you want to installing directly to Linux without Docker, you can install as follows "[2. Run on your environment without Docker](#2-run-on-your-environment-without-docker)".

If you want to install other way, you can refer to see install scripts in [install directory](https://github.com/ElliottTradeLaboratory/DQN/tree/master/install) for help to understanding how to install. 　Particularly, you need to install according to  _\<framework name\>_ _install.sh in install directory for frameworks. Because, The version of each framework is strictly specified for prevent the future releases of each framework from causing this DQN implementation to stop working.


### ◇Arcade Learning Environment

This implementation uses [alewrap_py](https://github.com/ElliottTradeLaboratory/alewrap_py) that reproduced [Deep Mind's alewrap](https://github.com/deepmind/alewrap) on python, but you can also use [Open AI's Gym](https://github.com/openai/gym).<br>
Since alewrap_py is submodules of this repository, you will be able to install it automatically, but Gym needs to be installed separately (but it is not required).<br>
However, the advantage of installing the gym is not only comparable with the alewrap_py version, but since all Atari game modules (e.g. breakout.bin) are bundled with atari_py that is bundled with Gym, There is also no need to search in to get them.<br>
If you follows to "[1. Run on Docler](#1-run-on-docker)" or "[2. Run on your environment without Docker](#2-run-on-your-environment-without-docker)", Gym will be automatically installed .

## Installation instructions

### 1. Run on Docker

#### 1-1. Install nvidia-docker

Install nvidia-docker as follows:
https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)

#### 1-2. Clone repository

```
cd <clone root dir>
git clone https://github.com/ElliottTradeLaboratory/DQN.git
```

NOTE: If you want see other source code such as alewrap_py and xitari, you can specify `--recursive` option and download them at same time.

#### 1-3. Build Docker images

Build Docker image(s) for any framework as follows:
```
$ cd <clone root dir>/DQN/install
$ ./build_docker_images <framework name> or all
　　・
　　・
　　・
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
pythorch            latest              xxxxxxxxxxxx        xxxxxxx             xxxxxx
tensorflow          latest              xxxxxxxxxxxx        xxxxxxx             xxxxxx
cntk                latest              xxxxxxxxxxxx        xxxxxxx             xxxxxx
mxnet               latest              xxxxxxxxxxxx        xxxxxxx             xxxxxx
$
```

#### 1-4. Run DQN

```
$ cd <clone root dir>/DQN
$ ./run_docker <framework name> [--logdir <log dir>]
mount /mnt/log_dir --> <log dir> if specified --logdir otherwise /tmp 
root@xxxxxx:/# cd DQM
root@xxxxxx:/DQN# ./run --backend <backend name> --env <game name> [options]
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
$ ./install_your_linux.sh <framework name> or all
```

#### 2-4. Run DQN

```
$ cd <clone root dir>/DQN
$ ./run --backend <backend name> --env <game name> [options]
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
`tensorflow` | Tensorflow-gpu 1.4.1 | 8.0 | 6.0
`cntk` | CNTK 2.3.1<br> Tensorflow 1.4.1(cpu) | 8.0 | 6.0
`mxnet` | MXNet 1.0.0<br> Tensorflow 1.2.1(cpu) | 8.0 | 5.1

* _\<game name\>_ is:<br>
　For alewrap_py,  a module name of the game exclude `.bin`(e.g. `breakout`).<br>
　For Gym, a env name of Gym(e.g. `Breakout-v0`).<br>
* _\<backend name\>_ as follows:

_\<backend name\>_ | Framework used for networks
---------------|----------
`pytorch` | PyTorch with torch.nn package
`pytorch_legacy` | PyTorch with torch.legacy.nn package
`tensorflow` | Tensorflow
`cntk` | CNTK
`mxnet` | MXNet



***
<a name="cntk_mpi"><sup>[1]</sup></a> https://docs.microsoft.com/en-us/cognitive-toolkit/Multiple-GPUs-and-machines<br>
<a name="pytorch_cuda"><sup>[2]</sup></a> Because it depends on Tensorflow that used for preprocessing and logging. It does not work on CUDA9.0 as of January 2018.<br>
<a name="tensorflow"><sup>[3]</sup></a> As of January 2018, if you want to run on Python3.6, you must install Tensorflow 1.3.0 as follows:<br>[for GPU]pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp36-cp36m-linux_x86_64.whl<br>[for CPU]pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-1.3.0-cp36-cp36m-linux_x86_64.whl.<br>
