#!/usr/bin/env bash
set -u
cd /data4/czh/yolov8-cluster/yolov8-hard-clustering || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
/data4/czh/anaconda3/envs/yolo_dg/bin/python -u train_style_adv.py --experiment latent_repeat --epochs 100 --batch_size 12 --seed 11 --num_workers 8 --results_path /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/comparison_20261005_100ep > /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/comparison_20261005_100ep/launch_logs/R3_latent_repeat.log 2>&1
training_exit=$?
printf '%s\n' "$training_exit" > /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/comparison_20261005_100ep/launch_logs/R3_latent_repeat.exit_code
exit "$training_exit"
