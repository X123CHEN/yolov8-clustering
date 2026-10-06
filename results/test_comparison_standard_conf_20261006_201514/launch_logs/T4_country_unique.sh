#!/usr/bin/env bash
set -u
cd /data4/czh/yolov8-cluster/yolov8-hard-clustering || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
/data4/czh/anaconda3/envs/yolo_dg/bin/python -u get_map_new.py --weights /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/comparison_20261005_100ep/country_unique/best_epoch_weights.pth --experiment country_unique --output /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/test_comparison_standard_conf_20261006_201514/country_unique > /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/test_comparison_standard_conf_20261006_201514/launch_logs/T4_country_unique.log 2>&1
eval_exit=$?
printf '%s\n' "$eval_exit" > /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/test_comparison_standard_conf_20261006_201514/launch_logs/T4_country_unique.exit_code
exit "$eval_exit"
