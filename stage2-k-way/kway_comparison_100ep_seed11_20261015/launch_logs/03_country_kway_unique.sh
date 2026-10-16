#!/usr/bin/env bash
set -e
cd /data4/czh/yolov8-cluster/yolov8-hard-clustering
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MPLBACKEND=Agg
exec /data4/czh/anaconda3/envs/yolo_dg/bin/python -u /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_100ep_seed11_20261015_150504/source_snapshot/run_experiment.py new_experiments.kway_domain.worker --suite /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_100ep_seed11_20261015_150504 --experiment 03_country_kway_unique --gpu 2 >> /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_100ep_seed11_20261015_150504/launch_logs/03_country_kway_unique.log 2>&1
