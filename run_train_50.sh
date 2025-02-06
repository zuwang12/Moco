#!/bin/bash

# Configuration for problem_size 50
PROBLEM_SIZE=50
TASK_BATCH_SIZE=32
MAX_LENGTH=100
CAUSAL="--causal"
BASELINE="--baseline avg"
PARALLEL_TASKS_TRAIN=128
OUTER_LR=1e-3
OUTER_TRAIN_STEPS=3000
PARALLEL_TASKS_VAL=128
VAL_PATH="data/tsp/val-50-coords.npy"
MODEL_SAVE_PATH="checkpoints/tsp50"

# Running the training script for problem_size 50
nohup python experiments/tsp_meta_train.py \
  --problem_size $PROBLEM_SIZE \
  --task_batch_size $TASK_BATCH_SIZE \
  --max_length $MAX_LENGTH \
  $CAUSAL \
  $BASELINE \
  --parallel_tasks_train $PARALLEL_TASKS_TRAIN \
  --outer_lr $OUTER_LR \
  --outer_train_steps $OUTER_TRAIN_STEPS \
  --parallel_tasks_val $PARALLEL_TASKS_VAL \
  --val_path $VAL_PATH \
  --model_save_path $MODEL_SAVE_PATH \
  > "./logs/train_tsp_50.log" 2>&1 &
