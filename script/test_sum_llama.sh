#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  



OUT_PATH=./log
dataset=xsum

python value.py \
    --prompt  "Identify the main idea or central theme of the text." \
    --dataset $dataset \
    --task sum \
    --iter 3 \
    --ad_iter 5 \
    --language_model llama\
    --setting default \
    --attack_type mix \
    --data_gen False \

python value.py -p $OUT_PATH > $OUT_PATH/result.txt
