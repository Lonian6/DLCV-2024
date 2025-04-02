#!/bin/bash

# TODO - run your inference Python3 code
python3 ./P3_inference.py --json $1 --save_dir $2 --ckpt $3 --config ./stable-diffusion/configs/stable-diffusion/v1-inference.yaml
