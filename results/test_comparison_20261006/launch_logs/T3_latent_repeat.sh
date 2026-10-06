#!/usr/bin/env bash
set -u
cd /data4/czh/yolov8-cluster/yolov8-hard-clustering || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
/data4/czh/anaconda3/envs/yolo_dg/bin/python -u get_map_new.py --weights /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/comparison_20261005_100ep/latent_repeat/best_epoch_weights.pth --experiment latent_repeat --output /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/test_comparison_20261006/latent_repeat > /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/test_comparison_20261006/launch_logs/T3_latent_repeat.log 2>&1
eval_exit=$?
printf '%s\n' "$eval_exit" > /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/test_comparison_20261006/launch_logs/T3_latent_repeat.exit_code
exit "$eval_exit"
