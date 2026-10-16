#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量重算 metrics_summary.csv（无需重训）。

用法:
    python regenerate_metrics_summary.py
    python regenerate_metrics_summary.py --logs-root ./logs/olddata
    python regenerate_metrics_summary.py --loss-dir logs/run3/train1/loss_2026_05_31_02_15_21
"""
import argparse
import csv
import os
import sys
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dg_config import CFG
from utils.metrics_summary import generate_summary


def _parse_bool(s):
    return str(s).strip().lower() in ('1', 'true', 'yes', 'on')


def load_ref_from_summary(loss_dir):
    ref_path = os.path.join(loss_dir, 'metrics_summary.csv')
    if not os.path.isfile(ref_path):
        return {}
    with open(ref_path, newline='', encoding='utf-8') as f:
        row = next(csv.DictReader(f))
    shape = row.get('input_shape', '640,640').strip('"').split(',')
    return {
        'results_path': row.get('results_path', ''),
        'use_adv': _parse_bool(row.get('use_adv', 'false')),
        'align_type': row.get('align_type', 'none'),
        'fft_enabled': _parse_bool(row.get('fft_enabled', 'false')),
        'mixstyle_enabled': _parse_bool(row.get('mixstyle_enabled', 'false')),
        'phi': row.get('phi', 's'),
        'input_shape': [int(shape[0]), int(shape[1])],
        'num_classes': int(row.get('num_classes', 4)),
        'batch_size': int(row.get('batch_size', 4)),
        'num_train': int(row.get('num_train', 0)),
        'num_val': int(row.get('num_val', 0)),
        'planned_epochs': int(row.get('total_epochs', 0)) or None,
    }


def build_args(loss_dir, ref, cli):
    train_dir = os.path.dirname(loss_dir)
    results_path = ref.get('results_path') or os.path.relpath(train_dir, SCRIPT_DIR)
    if not str(results_path).startswith('.'):
        results_path = './' + str(results_path)

    weights = cli.weights_path or os.path.join(train_dir, 'best_epoch_weights.pth')
    return SimpleNamespace(
        results_path=results_path,
        results_path_rel=True,
        weights_path=weights,
        classes_path=cli.classes_path,
        fps_image=cli.fps_image,
        fps_interval=cli.fps_interval,
        phi=ref.get('phi', 's'),
        input_shape=ref.get('input_shape', [640, 640]),
        num_classes=ref.get('num_classes', 4),
        batch_size=ref.get('batch_size', 4),
        num_train=ref.get('num_train', 0),
        num_val=ref.get('num_val', 0),
        planned_epochs=ref.get('planned_epochs'),
        use_adv=ref.get('use_adv', CFG.use_adv),
        align_type=ref.get('align_type', CFG.align_type),
        fft_enabled=ref.get('fft_enabled', CFG.fft_enabled),
        mixstyle_enabled=ref.get('mixstyle_enabled', CFG.mixstyle_enabled),
        num_gpus=cli.num_gpus,
        map_threshold=cli.map_threshold,
        cpu=cli.cpu,
    )


def find_loss_dirs(logs_root):
    loss_dirs = []
    for root, dirs, files in os.walk(logs_root):
        if 'metrics_epoch.csv' in files:
            loss_dirs.append(root)
    return sorted(loss_dirs)


def main():
    parser = argparse.ArgumentParser(description='Regenerate metrics_summary.csv in batch')
    parser.add_argument('--logs-root', default='./logs', help='scan all loss_* under this root')
    parser.add_argument('--loss-dir', default=None, help='regenerate a single loss directory')
    parser.add_argument('--weights-path', default=None, help='override weights path')
    parser.add_argument('--classes-path', default='model_data/rdd_classes.txt')
    parser.add_argument('--fps-image', default='img/street.jpg')
    parser.add_argument('--fps-interval', type=int, default=100)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--map-threshold', type=float, default=0.45)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if args.loss_dir:
        targets = [os.path.abspath(args.loss_dir)]
    else:
        targets = find_loss_dirs(os.path.abspath(args.logs_root))

    if not targets:
        print('No metrics_epoch.csv found.')
        return

    ok, fail = 0, 0
    for loss_dir in targets:
        try:
            ref = load_ref_from_summary(loss_dir)
            summary_args = build_args(loss_dir, ref, args)
            generate_summary(loss_dir, summary_args)
            ok += 1
        except Exception as e:
            fail += 1
            print(f'[FAIL] {loss_dir}: {e}')

    print(f'Done. success={ok}, failed={fail}')


if __name__ == '__main__':
    main()
