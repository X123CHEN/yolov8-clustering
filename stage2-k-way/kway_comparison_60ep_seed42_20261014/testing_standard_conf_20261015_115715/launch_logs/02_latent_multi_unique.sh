#!/usr/bin/env bash
set -u
cd /data4/czh/yolov8-cluster/yolov8-hard-clustering || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
export PYTHONPATH=/data4/czh/yolov8-cluster/yolov8-hard-clustering
/data4/czh/anaconda3/envs/yolo_dg/bin/python -u /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/testing_standard_conf_20261015_115715/code/evaluate_standard.py --weights /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/02_latent_multi_unique/best_epoch_weights.pth --experiment 02_latent_multi_unique --seed 42 --output /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/testing_standard_conf_20261015_115715/smoke/02_latent_multi_unique --limit 1 > /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/testing_standard_conf_20261015_115715/launch_logs/02_latent_multi_unique.smoke.log 2>&1
smoke_exit=$?
if [ "$smoke_exit" -ne 0 ]; then
  printf "%s\n" "$smoke_exit" > /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/testing_standard_conf_20261015_115715/launch_logs/02_latent_multi_unique.exit_code
  exit "$smoke_exit"
fi
/data4/czh/anaconda3/envs/yolo_dg/bin/python -u /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/testing_standard_conf_20261015_115715/code/evaluate_standard.py --weights /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/02_latent_multi_unique/best_epoch_weights.pth --experiment 02_latent_multi_unique --seed 42 --output /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/testing_standard_conf_20261015_115715/02_latent_multi_unique > /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/testing_standard_conf_20261015_115715/launch_logs/02_latent_multi_unique.log 2>&1
eval_exit=$?
printf "%s\n" "$eval_exit" > /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/testing_standard_conf_20261015_115715/launch_logs/02_latent_multi_unique.exit_code
if [ "$eval_exit" -eq 0 ]; then
  /data4/czh/anaconda3/envs/yolo_dg/bin/python /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/testing_standard_conf_20261015_115715/code/summarize.py >> /data4/czh/yolov8-cluster/yolov8-hard-clustering/kway_comparison_60ep_seed42_20261014_195549/testing_standard_conf_20261015_115715/launch_logs/summary.log 2>&1
fi
exit "$eval_exit"
