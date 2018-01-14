#!/bin/bash

if [ -z "$1" ]; then
    echo [usage] ./install_fw.sh [pytorch, tensorflow, cntk, mxnet or all]
    exit 1
fi

echo *** Installing OpenAI Gym ***
git clone https://github.com/openai/gym.git
cd gym
pip3 install -e .[atari]

echo *** Installing alewrap_py ***
cd /
cd /DQN/alewrap_py
python3 setup.py build
python3 setup.py install
cd /usr/local/lib/python3.5/dist-packages/alewrap_py-1.0.0-py3.5.egg/alewrap_py/atari_roms
ls /usr/local/lib/python3.5/dist-packages/atari_py/atari_roms/*.bin | xargs -I{} ln -s {}

frameworks="pytorch cntk tensorflow mxnet"
for backend in $baskends; do
    if [ "$backend" = "$1" -o "$1" = "all" ]; then
        echo *** Installing ${frameworks} ***
        ./${backend}_install.sh
    fi
done
pip3 install Pillow matplotlib python3-tk
