#!/usr/bin/env bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/data4/czh/anaconda3/envs/yolo_dg/bin/python -u /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/test_comparison_20261006/finalize_tests.py > /data4/czh/yolov8-cluster/yolov8-hard-clustering/results/test_comparison_20261006/launch_logs/finalize.log 2>&1
