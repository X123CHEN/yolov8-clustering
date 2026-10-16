"""Launch the verified four-way comparison on GPUs 0, 1, 2, 3 in screen."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch
import torchvision
from utils.comparison_training import ROOT, EXPERT_ROOT, prepare_data, file_hash, atomic_json
from utils.comparison_evaluation import DOMAINS
from new_experiments.kway_domain.report import EXPERIMENTS, TITLES, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    suite = args.suite.resolve()
    verification = json.loads((suite / 'verification.json').read_text())
    if not verification['passed'] or not verification.get('smoke_passed'):
        raise RuntimeError('Real-gradient and four-mode smoke checks must pass before launch')
    if (suite / 'run_manifest.json').exists():
        raise FileExistsError('This suite has already been launched; use its per-run screen script to resume')
    if verification['seed'] != args.seed or verification['epochs'] != args.epochs:
        raise ValueError('Calibration recipe differs from training recipe')
    for directory in ('configs', 'launch_logs', 'source_snapshot'):
        (suite / directory).mkdir(exist_ok=True)
    runs, jobs = [], {}
    for gpu, (name, title) in enumerate(zip(EXPERIMENTS, TITLES)):
        kind = 'country' if 'country' in name else 'latent'
        single = 'kway' in name
        alpha = verification['calibrations'][kind]['alpha_domain'] if single else .7
        config = dict(experiment=name, title=title, epochs=args.epochs, seed=args.seed, batch_size=12,
            num_workers=8, gpu=gpu, data_protocol='unique', alpha_domain=alpha,
            architecture='single_kway' if single else 'expert_multi',
            domain_count=3 if kind == 'country' else 2, feature='backbone_feat3',
            fab_reference='utils.comparison_training.domain_triplet', grl=single,
            lr_horizon_epochs=args.epochs, model_initialization=str(ROOT / 'model_data/yolov8_s.pth'))
        atomic_json(suite / 'configs' / f'{name}.json', config)
        screen = f'kway{args.seed}_{gpu + 1}_{kind}_{"single" if single else "multi"}_{suite.name[-6:]}'
        command = [sys.executable, '-u', str(suite / 'source_snapshot/run_experiment.py'), 'new_experiments.kway_domain.worker',
                   '--suite', str(suite), '--experiment', name, '--gpu', str(gpu)]
        script = suite / 'launch_logs' / f'{name}.sh'
        log = suite / 'launch_logs' / f'{name}.log'
        script.write_text('#!/usr/bin/env bash\nset -e\ncd ' + shlex.quote(str(ROOT)) + '\n' +
            f'export CUDA_DEVICE_ORDER=PCI_BUS_ID\nexport CUDA_VISIBLE_DEVICES={gpu}\nexport OMP_NUM_THREADS=2\nexport MKL_NUM_THREADS=2\nexport OPENBLAS_NUM_THREADS=2\nexport MPLBACKEND=Agg\n' +
            'exec ' + shlex.join(command) + ' >> ' + shlex.quote(str(log)) + ' 2>&1\n')
        script.chmod(0o755)
        runs.append(dict(**config, screen=screen, command=command, script=str(script), log=str(log)))
        jobs[name] = dict(phase='pending', gpu=gpu, screen=screen)
    train_files = {}
    for kind in ('country', 'latent'):
        data = prepare_data(kind + '_unique')
        train_files[kind] = {key: dict(path=str(path), sha256=file_hash(path)) for key, path in zip(('train', 'validation'), data[1:3])}
    test_lists = {d: ROOT / 'datasets/RDD2022' / d / 'ImageSets/Main/test_all.txt' for d in DOMAINS}
    test_counts = {d: len(p.read_text().split()) for d,p in test_lists.items()}
    tracked_diff = subprocess.check_output(['git','diff','HEAD'], cwd=ROOT, text=True)
    (suite / 'git_diff.patch').write_text(tracked_diff)
    manifest = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), project=str(ROOT),
        python=sys.executable, seed=args.seed, epochs=args.epochs, primary_metric="seven_domain_mean_voc_map50", confidence_mode="max_class_sigmoid", runs=runs, gpu_mapping=[0,1,2,3],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        git_diff=str(suite / 'git_diff.patch'), source_snapshot=str(suite / 'source_snapshot'),
        environment=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,
                         cuda=torch.version.cuda, gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv'],text=True)),
        initial_weights=dict(path=str(ROOT / 'model_data/yolov8_s.pth'),sha256=file_hash(ROOT / 'model_data/yolov8_s.pth')),
        data_lists=train_files, cluster_labels=dict(path=verification['labels']['artifact'],sha256=file_hash(verification['labels']['artifact'])),
        expert_root=str(EXPERT_ROOT), test_images=sum(test_counts.values()), test_counts=test_counts,
        test_lists={d:dict(path=str(p),sha256=file_hash(p)) for d,p in test_lists.items()},
        checkpoint_selection='minimum source validation detection loss',
        calibration='fixed source-only feature-gradient matching; see verification.json',
        timing='wall-clock training includes setup/validation/AP/save; testing includes load/inference/metrics/save')
    atomic_json(suite / 'run_manifest.json', manifest)
    atomic_json(suite / 'status.json', dict(complete=False, jobs=jobs))
    summarize(suite)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    gpu_states = {int(row.split(',')[0]): [int(v.strip()) for v in row.split(',')[1:]] for row in gpu_rows.splitlines()}
    for gpu in range(4):
        memory, utilization = gpu_states[gpu]
        if memory > 100 or utilization > 0:
            raise RuntimeError(f'GPU {gpu} is occupied: {memory} MiB, {utilization}%')
    for run in runs:
        subprocess.run(['screen','-dmS',run['screen'],'bash',run['script']], cwd=ROOT, check=True)
        print('STARTED',run['screen'],'GPU',run['gpu'],run['experiment'],flush=True)
    print('SUITE',suite,flush=True)


if __name__ == '__main__':
    main()
