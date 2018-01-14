#!/bin/bash

if [ -z "$1" ]; then
    echo [usage] ./install.sh [pytorch, tensorflow, cntk, mxnet or all]
    exit 1
fi
baskends="pytorch cntk tensorflow mxnet"
for backend in $baskends; do
    if [ "$backend" = "$1" -o "$1" = "all" ]; then
        ./${backend}_install.sh
        #pip3 install sk-video opencv-python Pillow
    fi
done
