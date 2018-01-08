#!/bin/sh
echo == backup source ===
ROOTDIR=$1
LOGDIR=$2
zip -r $LOGDIR/source_dqn.zip $ROOTDIR/dqn/*.py
zip -r $LOGDIR/source_dqn.zip $ROOTDIR/dqn/test/*.py
zip -r $LOGDIR/source_utils.zip $ROOTDIR/utils/*.py
zip -r $LOGDIR/source_utils.zip $ROOTDIR/utils/test/*.py
zip -r $LOGDIR/source_alewrap_py.zip $ROOTDIR/alewrap_py/*.py
zip -r $LOGDIR/source_alewrap_py.zip $ROOTDIR/alewrap_py/test/*.py
