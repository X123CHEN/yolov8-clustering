"""Compare frozen training entries using three real batches and a common recipe."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import importlib.util
from pathlib import Path
import sys

PROJECT = Path('/data4/czh/yolov8-cluster/yolov8-hard-clustering')
SUITE = PROJECT/'kway_comparison_60ep_seed42_20261014_195549'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['first', 'second'], required=True)
parser.add_argument('--domain', choices=['country', 'latent'], required=True)
args = parser.parse_args()

import torch
torch.use_deterministic_algorithms(True)
backend_type = type(torch.backends.cudnn)
backend_type.benchmark = property(lambda self: False, lambda self, value: None)
backend_type.deterministic = property(lambda self: True, lambda self, value: None)


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


snapshot = (PROJECT/'results/comparison_20261005_100ep/source_snapshot' if args.stage=='first'
            else SUITE/'source_snapshot')
# Load archived definitions while keeping the actual data and expert paths.
for name in ['nets.yolo_training', 'nets.maddg', 'utils.dataloader', 'utils.callbacks']:
    path = snapshot/Path(*name.split('.')).with_suffix('.py')
    load(name, path)
helper = load('utils.comparison_training', snapshot/'utils/comparison_training.py')
helper.ROOT = PROJECT
helper.EXPERT_ROOT = PROJECT.parent/'yolov8-pretrain/results/expert_pretrain_20261004_200ep'
if args.stage == 'first':
    entry = helper
    experiment = args.domain+'_unique'
else:
    entry = load('parity_training', snapshot/'new_experiments/kway_domain/training.py')
    experiment = ('01_country' if args.domain=='country' else '02_latent')+'_multi_unique'
save = entry.atomic_save


def compact_save(value, path):
    if path.name == 'training_state.pt':
        value = {k: value[k] for k in ['model', 'ema', 'discriminators']}
    save(value, path)


entry.atomic_save = compact_save
sys.argv = ['training', '--experiment', experiment, '--results_path', str(OUT/('parity_'+args.stage)),
            '--epochs', '100', '--seed', '11', '--batch_size', '12', '--num_workers', '8', '--smoke_steps', '3']
entry.main()
