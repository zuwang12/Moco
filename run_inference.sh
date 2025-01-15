#!/bin/bash

# 경로 설정 (사용자의 경로로 수정)
DATA_PATH="./data/tsp/test-100-sample.npy"
CHECKPOINT_FOLDER="./checkpoints/tsp100_200_32"
num_cities=100
constraint_type="basic"

# 평가 스크립트 실행
nohup python experiments/evaluate_tsp.py \
    --data_path "$DATA_PATH" \
    --task_batch_size 32 \
    --batch_size_eval 1 \
    --num_steps 200 \
    --num_starting_nodes 32 \
    --checkpoint_folder "$CHECKPOINT_FOLDER" \
    --num_cities $num_cities \
    --constraint_type $constraint_type \
    > "./logs/inference_test_${num_cities}_${constraint_type}.log" 2>&1 &
