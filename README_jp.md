# PyTorch、MXNet、TensorflowおよびCNTKを使用した、PythonによるDQN3.0レベルのパフォーマンスを有するDQN実装

このリポジトリにはPyTorch, MXNet, Tensorflow and CNTK(両者はKeras上で)といったメジャーなDeep Learningフレームワークを使用したDeep Q-Network[(Mnih et al., 2015)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)(DQN)の実装が含まれています。<br>
これらの実装方法とパフォーマンスは、[Mnih et al., [2015]](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)の参照実装である[DQN3.0](https://github.com/deepmind/dqn)と全く同じです.

詳細は[wiki](https://github.com/ElliottTradeLaboratory/DQN/wiki)を参照してください。

## インストール概要

* インストールにはLinuxが必須です。
* GPUの仕様を強く推奨します。　なぜならば、DQN3.0に最も近い設定でもっとも高速なバージョンであっても、GTX1080it上で実行した場合500万ステップの学習に約7日間を要すからです。

### Deep Learningフレームワーク

この実装は複数のフレームワークを使用していますが、それらすべてをインストールすることも、または1つのフレームワークのみをインストールすることも可能です。　しかし、Tensorflowはロギングのために必須です。

最も簡単にあなたのLinux環境を壊すことなく実行させる方法は、"[1. Docker上で実行](#1-run-on-docker)"に従い、[Dockerfiles](https://github.com/ElliottTradeLaboratory/DQN/tree/master/install)を使用し、[Docker](https://www.docker.com/)上に環境を構築することです。

特に、CNTKの場合、異なる引数を指定したトレイニングプロセスを並行に実行させたい時はDockerを使うと便利です。　なぜならば、CNTKの並行処理はMPIで実装されているため、プロセスを実行する際には`mpiexec`を使用して`mpiexec --npernode $num_workers python training.py`というように実行させなければならないからです[<sup>[1]</sip>](#cntk_mpi)。

