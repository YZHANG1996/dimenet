#!/bin/sh

APPDIR=`dirname $0`
cd ./repo
pip list
pip install numpy==1.19.2
python train.py

# python -u $APPDIR/main_qm9.py --num_workers 4 $@
# return $?
