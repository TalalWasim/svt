CUDA_LAUNCH_BLOCKING=1

PROJECT_PATH="./"
DATA_PATH="../datasets/kinetics-dataset/k400_resized/annotations_svt"
EXP_NAME="test"

cd "$PROJECT_PATH" || exit

if [ ! -d "./results/$EXP_NAME" ]; then
  mkdir "./results/$EXP_NAME"
fi

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port="$RANDOM" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 6 \
  --data_path "${DATA_PATH}" \
  --output_dir "./results/$EXP_NAME" \
  --num_workers 6 \
  --epochs 30 \
  --use_fp16 False \
  --opts \
  DATA.NUM_FRAMES 16 \
  MODEL.TUBELET_SIZE 2 \
  DATA.RAND_FR True \
  NUM_GPUS 2 \
  DATA_LOADER.NUM_WORKERS 6 \
  TIMESFORMER.PRETRAINED_MODEL '../pretrained/svt_vmae.pth'