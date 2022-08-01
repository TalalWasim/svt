PROJECT_PATH="./"
DATA_PATH="../datasets/kinetics-dataset/k400_resized"
DATASET="kinetics400"

EXP_NAME="svt_dino"
CHECKPOINT="results/$EXP_NAME/checkpoint.pth"

cd "$PROJECT_PATH" || exit

if [ ! -d "eval/$EXP_NAME" ]; then
  mkdir "eval/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_linear.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 20 \
  --lr 0.001 \
  --batch_size_per_gpu 8 \
  --num_workers 4 \
  --num_labels 400 \
  --dataset "$DATASET" \
  --output_dir "eval/$EXP_NAME" \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/annotations_svt" \
  DATA.USE_FLOW False