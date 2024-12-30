#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  



OUT_PATH=./log
dataset=asset
python run.py \
    --prompt  "Please summary the main context." \
    --dataset $dataset \
    --task sim \
    --language_model gpt\
    --llm_type turbo \
    --sample_num default \
    --setting default \
    --iter 3 \
    --ad_iter 5 \
    --data_gen False \
    --attack_type mix \


python run.py -p $OUT_PATH > $OUT_PATH/result.txt
