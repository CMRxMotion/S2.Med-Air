#!/usr/bin/env sh
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun -p smart_health_00  -N 1  --gres=gpu:1 --cpus-per-task=10 python inference.py --input=/mnt/lustre/gongshizhan/data_and_label/challen_data/dataset/nnUnet_raw/nnUNet_raw_data/Task001_heart/imagesTs/ --output=/mnt/lustre/gongshizhan/test  2>&1|tee log/train-$now.log &
