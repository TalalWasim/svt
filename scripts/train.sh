#!/bin/bash

PROJECT_PATH="./"
DATA_PATH="../datasets/kinetics_dataset/k400_resized/annotations_new"
EXP_NAME="args_test"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

python -m torch.distributed.launch \
  --nproc_per_node=16 \
  --master_port="$RANDOM" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 8 \
  --data_path "${DATA_PATH}" \
  --output_dir "checkpoints/$EXP_NAME" \
  --opts \
  MODEL.TWO_STREAM False \
  MODEL.TWO_TOKEN False \
  DATA.NO_FLOW_AUG False \
  DATA.USE_FLOW False \
  DATA.RAND_CONV False \
  DATA.NO_SPATIAL False \
  DATA.RAND_FR True \
  SOLVER.MAX_EPOCH 20

