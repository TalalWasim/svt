#!/bin/bash
#SBATCH --job-name=mini_vmae_30_tube
#SBATCH --partition=multigpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:16

PROJECT_PATH="./"
DATA_PATH="../datasets/kinetics-dataset/k400_resized_2/annotations_mini"
EXP_NAME="k400_mini/svt_vmae_30_tube_2"

cd "$PROJECT_PATH" || exit

if [ ! -d "./results/$EXP_NAME" ]; then
  mkdir "./results/$EXP_NAME"
fi

python -m torch.distributed.launch \
  --nproc_per_node=16 \
  --master_port="$RANDOM" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 6 \
  --data_path "${DATA_PATH}" \
  --output_dir "./results/$EXP_NAME" \
  --num_workers 6 \
  --epochs 30 \
  --weight_decay 0.04 \
  --weight_decay_end 0.1 \
  --use_fp16 False \
  --opts \
  NUM_GPUS 16 \
  DATA_LOADER.NUM_WORKERS 6 \
  DATA.RAND_FR True \
  DATA.NUM_FRAMES 16 \
  MODEL.TUBELET_SIZE 2 \
  TIMESFORMER.PRETRAINED_MODEL '../pretrained/svt_vmae.pth'
