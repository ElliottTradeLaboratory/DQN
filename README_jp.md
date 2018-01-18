# PyTorch、MXNet、TensorflowおよびCNTKを使用した、PythonによるDQN3.0レベルのパフォーマンスを有するDQN実装

このリポジトリにはPyTorch, MXNet, Tensorflow and CNTK(両者はKeras上で)といったメジャーなDeep Learningフレームワークを使用したDeep Q-Network[(Mnih et al., 2015)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)(DQN)の実装が含まれています。<br>
これらの実装方法とパフォーマンスは、[Mnih et al., [2015]](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)の参照実装である[DQN3.0](https://github.com/deepmind/dqn)と全く同じです.

詳細は[wiki](https://github.com/ElliottTradeLaboratory/DQN/wiki/ホーム)を参照してください。

## インストール概要

* インストールにはLinuxが必須です。
* GPUの仕様を強く推奨します。　なぜならば、DQN3.0に最も近い設定でもっとも高速なバージョンであっても、GTX1080it上で実行した場合500万ステップの学習に約7日間を要すからです。

### Deep Learningフレームワーク

この実装は複数のフレームワークを使用していますが、それらすべてをインストールすることも、または1つのフレームワークのみをインストールすることも可能です。　しかし、Tensorflowはロギングのために必須です。

最も簡単にあなたのLinux環境を壊すことなく実行させる方法は、「[1. Dockerで実行する場合](#1-run-on-docker)」に従い、[Dockerfiles](https://github.com/ElliottTradeLaboratory/DQN/tree/master/install)を使用し、[Docker](https://www.docker.com/)上に環境を構築することです。

特に、CNTKの場合、異なる引数を指定したトレイニングプロセスを並行に実行させたい時はDockerを使うと便利です。　なぜならば、CNTKの並行処理はMPIで実装されているため、プロセスを実行する際には`mpiexec`を使用して`mpiexec --npernode $num_workers python training.py`というように実行させなければならないからです[<sup>[1]</sip>](#cntk_mpi)。

もしDockerを使用せずLinuxに直接インストールしたい場合は、「[2. Docker以外の環境で実行する場合](#2-run-on-your-environment-without-docker)」に従ってインストールすることができます。

もし上記以外の方法でインストールしたい場合、[installディレクトリ](https://github.com/ElliottTradeLaboratory/DQN/tree/master/install)にあるインストール・スクリプトファイルが、どのようにインストールしたらよいかを理解するための参考になります。　特に、フレームワークをインストールする際には、installディレクトリにある _\<framework name\>_ _install.shに従いインストールする必要があります。　なぜならば、このDQN実装がフレームワークの将来のバージョンの影響により動作しなくなることを防ぐために、各フレームワークのバージョンは厳密に指定しているからです。

### ◇Arcade Learning Environment

この実装では、[Deep Mindのxitari](https://github.com/deepmind/xitari)のforkである[alewrap_py](https://github.com/ElliottTradeLaboratory/alewrap_py)を使用していますが、[Open AIのGym](https://github.com/openai/gym)も使用することができます。　alewrap_pyとxitariはこのモジュールのサブモジュールなので、それらを自動的にインストールすることが可能ですが、Gymはそれとは別にインストールする必要があります（しかし、必須ではありません）。<br>
しかし、Gymをインストールすることのアドバンテージは、alewrap_py版との比較ができることだけでなく、Gymに同梱されているatari-pyが、全てのAtariゲームモジュールを同梱しているため、それらのモジュールをインターネット上で探す必要が無くなるということです。<br>
もし「[1. Dockerで実行する場合](#1-run-on-docker)」または「[2. Docker以外の環境で実行する場合](#2-run-on-your-environment-without-docker)」に従ってインストールすると、Gymは自動的にインストールされます。

## Installation手順

### 1. Dockerで実行する場合

#### 1-1. Install nvidia-docker

次の手順に従い、nvidia-dockerをインストールします:
https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)

## 1-2. リポジトリのクローン

```
cd <clone root dir>
git clone https://github.com/ElliottTradeLaboratory/DQN.git
```

注：もしalewrap_pyやxitariのソースコードも見たい場合は、`--recursive`を指定することでそれらを同時にダウンロードすることもできます。


#### 1-3. Docler imageのビルド

各フレームワーク用のDocker imageを次の手順に従いビルドします:
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

#### 1-4. DQNの実行

```
$ cd <clone root dir>/DQN
$ ./run_docker <framework name> [--logdir <log dir>]
mount /mnt/log_dir --> <log dir> if specified --logdir otherwise /tmp 
root@xxxxxx:/# cd DQM
root@xxxxxx:/DQN# ./run --backend <backend name> --env <game name> [options]
```

### 2. Run on your environment without Docker

#### 2-1. Install CUDA and CUDNN

GPUを使用する場合は、次のサイトに従いCUDAとcuDNNをインストールする必要があります:

CUDA  : ダウンロード: https://developer.nvidia.com/cuda-toolkit-archive<br>
　　　　インストールガイド: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html<br>
cuDNN : ダウンロー: https://developer.nvidia.com/rdp/cudnn-download<br>
　　　　インストールガイド: [cuDNN5.1](http://developer2.download.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/doc/cudnn_install.txt?4Y7u0FqHrotFcmVuCKOpM2anE-n8iMSBbn9WCrSMFTUFQzXCSGfEIkdPvFi0yoyTYBTKJzIiKiVwvgSYDqnfDzpew8WT1PdIAnXOeStXoMX2meBxzvBWZmNaVc3dt5u8Cv96mWCoTVp87ppWFM22UG1vqwAgwu4pR-W7m7fuHGOfIMYr), [cuDNN6.0](http://developer2.download.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/Doc/cudnn_install-2.txt?5e1fCcgO0eYlHY7zwZH-LBiJJBZRX4pF_wv1Gf3hq1lpsF6Q0pvkc0BkdZKVwfxaT-m8iAjLn0ZV6NRh_-jGp8GCMDnmUmCHtxQ82UQnwQVlrzZebTFGRm5q90Ic8S7UC2SMG0Z-NXlwLQfqOpr7l6YErWhJB1Ai2dc4ggsXjPFAtEx_)

注）CUDAとcuDNNのバージョンは使用するフレームワークのバージョンに依存します。 詳しくは[説明](#description)をご覧ください。

#### 2-2. リポジトリのクローン

```
$ cd <clone root dir>
$ git clone --recursive https://github.com/ElliottTradeLaboratory/DQN.git
```

#### 2-3. インストールスクリプトの実行

```
$ cd <clone root dir>/DQN/install
$ ./install_your_linux.sh <framework name> or all
```
