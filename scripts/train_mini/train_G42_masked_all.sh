#!/bin/bash
#SBATCH --job-name=mini_mask_all_30
#SBATCH --partition=multigpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:16

PROJECT_PATH="./"
DATA_PATH="../datasets/kinetics-dataset/k400_resized_3/annotations_mini"
EXP_NAME="k400_mini/svt_masked_all_30_mae_vmae"

cd "$PROJECT_PATH" || exit

if [ ! -d "./results/$EXP_NAME" ]; then
  mkdir "./results/$EXP_NAME"
fi

python -m torch.distributed.launch \
  --nproc_per_node=16 \
  --master_port="$RANDOM" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 1 \
  --data_path "${DATA_PATH}" \
  --output_dir "./results/$EXP_NAME" \
  --num_workers 6 \
  --epochs 30 \
  --use_fp16 False \
  --opts \
  MODEL.MASKED True \
  MODEL.JOINT_MASK_CROP False \
  MODEL.JOINT_MASK_ONLY True \
  DATA.RAND_FR True \
  NUM_GPUS 16 \
  DATA_LOADER.NUM_WORKERS 6 \
  TIMESFORMER.PRETRAINED_MODEL '../pretrained/enc_mae_dec_vmae.pth'