#!/bin/bash

# 경로 설정 (사용자의 경로로 수정)
num_cities=500
constraint_type="basic"
CHECKPOINT_FOLDER="./checkpoints/tsp${num_cities}_200_32"
k=$num_cities
batch_size=32
top_k=32
num_starting_nodes=2
num_steps=200
two_opt_t_max=None
now=$(date +"%F_%T")
DATA_PATH="./data/tsp/test-500-coords.npy"
testYn='N'

# 로그 디렉토리 생성
LOG_DIR="./logs/${constraint_type}"
mkdir -p "$LOG_DIR"

# 평가 스크립트 실행
nohup python experiments/evaluate_tsp.py \
    --data_path "$DATA_PATH" \
    --task_batch_size $batch_size \
    --batch_size_eval $batch_size \
    --num_steps $num_steps \
    --num_starting_nodes $num_starting_nodes \
    --checkpoint_folder $CHECKPOINT_FOLDER \
    --num_cities $num_cities \
    --constraint_type $constraint_type \
    --run_time $now \
    --k $k \
    --top_k $top_k \
    --testYn $testYn \
    > "$LOG_DIR/moco_inference_${num_cities}_${constraint_type}_${now}.log" 2>&1 &
