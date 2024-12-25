#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  



OUT_PATH=./log
dataset=sam
python run.py \
    --prompt  "Please summary the main context." \
    --dataset $dataset \
    --task sum \
    --language_model gpt\
    --llm_type turbo \
    --sample_num default \
    --setting default \
    --iter 3 \
    --ad_iter 5 \
    --data_gen False \
    --attack_type combine \


python run.py -p $OUT_PATH > $OUT_PATH/result.txt
