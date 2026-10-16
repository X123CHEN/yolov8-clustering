#!/usr/bin/env bash
set -e
cd /data4/czh/yolov8-cluster/yolov8-hard-clustering
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /data4/czh/anaconda3/envs/yolo_dg/bin/python -u -m new_experiments.kway_domain.worker --suite /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549 --experiment 02_latent_multi_unique --gpu 1 >> /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/launch_logs/02_latent_multi_unique.log 2>&1
