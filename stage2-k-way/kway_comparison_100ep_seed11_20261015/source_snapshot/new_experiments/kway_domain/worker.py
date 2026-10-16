"""One GPU/screen per experiment: train, test seven domains, and update reports."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from utils.comparison_training import ROOT, atomic_json
from new_experiments.kway_domain.report import summarize, EXPERIMENTS


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--gpu', choices=('0','1','2','3'), required=True)
    args = parser.parse_args()
    suite, name = args.suite.resolve(), args.experiment
    folder = suite / name
    folder.mkdir(exist_ok=True)
    config = json.loads((suite / 'configs' / f'{name}.json').read_text())
    verified = json.loads((suite / 'verification.json').read_text())
    if not verified['passed']: raise RuntimeError('Pre-training verification did not pass')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
    timing = json.loads((folder / 'timing.json').read_text()) if (folder / 'timing.json').exists() else {}

    def update(phase, **extra):
        with (suite / 'report.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads((suite / 'status.json').read_text())
            state['jobs'][name].update(phase=phase, gpu=int(args.gpu), pid=os.getpid(),
                screen=os.environ.get('STY'), updated_at=stamp(), **extra)
            state['complete'] = all(job['phase'] == 'complete' for job in state['jobs'].values())
            atomic_json(suite / 'status.json', state)
            summarize(suite)
        print(name, phase, extra, flush=True)

    def run(phase, command):
        update(phase, started_at=stamp())
        start = time.perf_counter()
        print('COMMAND', command, flush=True)
        with (folder / f'{phase}.log').open('ab', buffering=0) as log:
            log.write((f'\nSTART {stamp()} {command!r}\n').encode())
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(folder / 'process.json', dict(phase=phase, child_pid=process.pid, worker_pid=os.getpid(), gpu=args.gpu))
            code = process.wait()
        timing[f'{phase}_s'] = timing.get(f'{phase}_s', 0.) + time.perf_counter() - start
        timing[f'{phase}_finished_at'] = stamp()
        atomic_json(folder / 'timing.json', timing)
        if code: raise subprocess.CalledProcessError(code, command)

    try:
        if not (folder / 'completion.json').exists():
            command = [sys.executable, '-u', str(Path(__file__).resolve().parents[2] / 'run_experiment.py'), 'new_experiments.kway_domain.training',
                '--experiment', name, '--results_path', str(suite), '--epochs', str(config['epochs']),
                '--seed', str(config['seed']), '--batch_size', str(config['batch_size']),
                '--num_workers', str(config['num_workers']), '--alpha_domain', str(config['alpha_domain'])]
            if (folder / 'training_state.pt').exists(): command.append('--resume')
            run('training', command)
        done = json.loads((folder / 'completion.json').read_text())
        if done['smoke_test'] or done['epochs_completed'] != config['epochs']:
            raise RuntimeError('Requested training epochs are incomplete')
        update('training_complete')
        evaluation = folder / 'testing'
        if not (evaluation / 'completion.json').exists():
            run('testing', [sys.executable, '-u', str(Path(__file__).resolve().parents[2] / 'run_experiment.py'), 'new_experiments.kway_domain.evaluation',
                '--weights', str(folder / 'best_epoch_weights.pth'), '--output', str(evaluation),
                '--experiment', name, '--seed', str(config['seed'])])
        done = json.loads((evaluation / 'completion.json').read_text())
        expected = json.loads((suite / 'run_manifest.json').read_text())['test_images']
        if done['smoke_test'] or done['domains_completed'] != 7 or done['images'] != expected:
            raise RuntimeError('Seven-domain test coverage is incomplete')
        update('complete')
    except BaseException as exc:
        update('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
