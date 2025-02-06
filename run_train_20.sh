#!/bin/bash

# Configuration for problem_size 20
PROBLEM_SIZE=20
TASK_BATCH_SIZE=32
MAX_LENGTH=50
CAUSAL="--causal"
BASELINE="--baseline avg"
PARALLEL_TASKS_TRAIN=64  # Smaller problem size, less parallelism
OUTER_LR=1e-3
OUTER_TRAIN_STEPS=6000  # Fewer steps, smaller problem
PARALLEL_TASKS_VAL=64  # Corresponding validation tasks
VAL_PATH="data/tsp/val-20-coords.npy"
MODEL_SAVE_PATH="checkpoints/tsp20"

# Running the training script for problem_size 20
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
  > "./logs/train_tsp_20.log" 2>&1 &
