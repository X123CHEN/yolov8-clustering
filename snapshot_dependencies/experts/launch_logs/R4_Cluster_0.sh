#!/usr/bin/env bash
set -u
cd /data4/czh/yolov8-cluster/yolov8-pretrain || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
/data4/czh/anaconda3/envs/yolo_dg/bin/python -u train_style_single.py --expert Cluster_0 --epochs 200 --batch_size 16 --manual_seed 11 --num_workers 8 --results_path /data4/czh/yolov8-cluster/yolov8-pretrain/results/expert_pretrain_20261004_200ep > /data4/czh/yolov8-cluster/yolov8-pretrain/results/expert_pretrain_20261004_200ep/launch_logs/R4_Cluster_0.log 2>&1
training_exit=$?
printf '%s\n' "$training_exit" > /data4/czh/yolov8-cluster/yolov8-pretrain/results/expert_pretrain_20261004_200ep/launch_logs/R4_Cluster_0.exit_code
exit "$training_exit"
