#!/bin/bash

echo *** Installing OpenAI Gym ***
git clone https://github.com/openai/gym.git
cd gym
pip install -e .[atari]

echo *** Installing alewrap_py ***
cd /
cd /DQN/alewrap_py
python3 setup.py build
python3 setup.py install

PACKAGES_DIR=`python3 /DQN/install/get_site_packages_path.py`
ALEWRAP_PY_DIR=`python3 /DQN/install/get_site_packages_path.py --get_alewrap_dir`
cd ${ALEWRAP_PY_DIR}/alewrap_py/atari_roms
ls ${PACKAGES_DIR}/atari_py/atari_roms/*.bin | xargs -I{} ln -s {}

cd /DQN/install
for framework in $@; do
    echo *** Installing ${framework} ***
    ./${framework}_install.sh
done
pip install Pillow matplotlib
