#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  



OUT_PATH=./log
dataset=sam

python data_generate.py \
    --prompt  "Please summary the main context" \
    --dataset $dataset \
    --task sum \
    --iter 3 \
    --ad_iter 5 \
    --language_model gpt\
    --setting default \
    --attack_type mix \
    --data_gen False \

python value.py -p $OUT_PATH > $OUT_PATH/result.txt
