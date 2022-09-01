#!/bin/sh

APPDIR=`dirname $0`
cd ./repo
pip install sympy
pip install tensorflow-addons==0.17.1
pip install numpy==1.19.2
python train.py $@

return $?

# python -u $APPDIR/main_qm9.py --num_workers 4 $@
# return $?
