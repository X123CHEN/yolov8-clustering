"""One experiment per screen: train/resume, evaluate, coordinate the final report."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latent_dg_config as settings
from tools.run_latent_suite import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--experiment', required=True, choices=settings.EXPERIMENTS)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    suite, name = args.suite.resolve(), args.experiment
    # The lock spans the complete worker lifetime, including evaluation.
    worker_lock = (suite / f'{name}.worker.lock').open('a')
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = settings.validate(json.loads((suite / f'{name}.json').read_text()))
    initial_status = json.loads((suite / 'status.json').read_text())
    verification = json.loads(Path(initial_status['verification']).read_text())
    if not verification['passed']:
        raise ValueError('Implementation verification is not complete')
    for relative, digest in verification['source_sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Implementation changed since verification: {relative}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    training, evaluation = suite / 'training' / name, suite / 'evaluation' / name

    def update(phase, **extra):
        with (suite / 'status.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads((suite / 'status.json').read_text())
            job = state['jobs'][name]
            job.update(phase=phase, gpu=args.gpu, pid=os.getpid(), screen=os.environ.get('STY'), **extra)
            state['screen_sessions'][name] = os.environ.get('STY')
            if phase in ('evaluation', 'complete') and name not in state['finished_training']:
                state['finished_training'].append(name)
            if phase == 'complete':
                state['evaluations'][name] = str(evaluation)
            failed = any(j['phase'] == 'failed' for j in state['jobs'].values())
            state['status'] = 'running_with_failures' if failed else 'running'
            if all(j['phase'] == 'complete' for j in state['jobs'].values()):
                from utils.summarize_latent_ablation import summarize
                ordered = {key: state['evaluations'][key] for key in settings.EXPERIMENTS if key in state['evaluations']}
                summarize(ordered, suite / 'comparison')
                state.update(status='complete', current=None)
            elif all(j['phase'] in ('complete', 'failed') for j in state['jobs'].values()):
                state.update(status='failed', current=None)
            write_json(suite / 'status.json', state)
        print(name, phase, extra, flush=True)

    def run(command, log_path):
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        with log_path.open('ab', buffering=0) as log:
            marker = f'\nSCREEN_WORKER_LAUNCH {stamp} gpu={args.gpu} screen={os.environ.get("STY")} command={command!r}\n'
            log.write(marker.encode()); print(marker, flush=True)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, bufsize=0)
            for chunk in iter(lambda: process.stdout.read(8192), b''):
                log.write(chunk)
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)

    try:
        if not (training / 'completion.json').exists():
            if (training / 'run_config.json').exists():
                if not (training / 'training_state.pt').exists():
                    raise ValueError('Uncommitted first epoch must be archived before a fresh run')
                update('resuming_training')
                command = [sys.executable, '-u', str(ROOT / 'train_latent_ablation.py'), '--resume', str(training)]
            else:
                update('training')
                command = [sys.executable, '-u', str(ROOT / 'train_latent_ablation.py'), '--resolved-config',
                           str(suite / f'{name}.json'), '--results-path', str(training)]
            run(command, suite / f'{name}_train.log')
        completion = json.loads((training / 'completion.json').read_text())
        if completion['smoke_test'] or completion['epochs_completed'] != config['epochs']:
            raise ValueError('Incomplete requested training budget')
        update('evaluation')
        if not (evaluation / 'completion.json').exists():
            if (evaluation / 'evaluation_config.json').exists():
                raise ValueError('Incomplete evaluation must be archived before retry')
            run([sys.executable, '-u', str(ROOT / 'get_map_new.py'), '--weights', str(training / 'best_epoch_weights.pth'),
                 '--output', str(evaluation), '--experiment', name], suite / f'{name}_test.log')
        completion = json.loads((evaluation / 'completion.json').read_text())
        if completion['smoke_test'] or completion['domains_completed'] != 7 or completion['images'] != 9433:
            raise ValueError('Incomplete seven-domain evaluation')
        update('complete')
    except BaseException as exc:
        update('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
