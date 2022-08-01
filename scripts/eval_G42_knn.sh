#!/bin/bash
#SBATCH --job-name=svt_knn_eval_2
#SBATCH --partition=default-long
#SBATCH --time=24:0:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1



PROJECT_PATH="./"
DATA_PATH="../datasets/hmdb51"
DATASET="hmdb51"

EXP_NAME="svt_dino"
CHECKPOINT="results/$EXP_NAME/checkpoint.pth"

cd "$PROJECT_PATH" || exit

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_knn.py \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --batch_size_per_gpu 128 \
  --nb_knn 5 \
  --temperature 0.07 \
  --num_workers 6 \
  --dataset "$DATASET" \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/annotations_svt" \
  DATA.PATH_LABEL_SEPARATOR ' ' \
  DATA.DECODING_BACKEND 'torchvision'
