#!/bin/bash
# python3 .py $1 $2
python3 inference.py -m ./setting3 -s $1 --output_root $2
# $1 ./dataset/private_test
# TODO - run your inference Python3 code