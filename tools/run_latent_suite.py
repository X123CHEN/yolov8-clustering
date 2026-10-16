"""Freeze/resume a suite; sequential or one GPU per experiment under screen."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
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


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', choices=settings.EXPERIMENTS,
                        default=None)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume-suite', type=Path, help='Reuse frozen configs; resume completed epoch checkpoints')
    parser.add_argument('--gpus', nargs='+', help='One distinct physical GPU per experiment; enables parallel jobs')
    parser.add_argument('--verification', type=Path, default=ROOT / 'results/latent_ablation_validation/verification.json')
    args = parser.parse_args()
    if args.resume_suite:
        if args.output or args.experiments:
            parser.error('--resume-suite cannot combine with --output/--experiments')
        saved_state = json.loads((args.resume_suite / 'status.json').read_text())
        args.experiments = saved_state['experiments']
    elif args.experiments is None:
        args.experiments = ['soft_expert', 'hard_kway', 'soft_kway']
    if args.gpus and (len(args.gpus) != len(args.experiments) or len(set(args.gpus)) != len(args.gpus)):
        parser.error('--gpus must provide one distinct GPU per experiment')
    if len(set(args.experiments)) != len(args.experiments):
        parser.error('Duplicate experiments')
    verification = json.loads(args.verification.read_text())
    if not verification['passed']:
        raise ValueError('Migration/smoke verification must pass before formal suite')
    for relative, digest in verification['source_sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Implementation changed since verification: {relative}')
    if args.resume_suite:
        configs = {name: settings.validate(json.loads((args.resume_suite / f'{name}.json').read_text())) for name in args.experiments}
        evaluations = dict(saved_state['evaluations'])
    else:
        configs = {name: settings.resolve(name) for name in args.experiments}
        evaluations = {}
    if 'hard_expert' not in configs and 'hard_expert' not in evaluations:
        historical = ROOT / 'results/comparison_20261005_100ep/latent_unique'
        old = json.loads((historical / 'run_config.json').read_text())
        baseline = settings.resolve('hard_expert')
        correspondence = {'epochs':'epochs', 'seed':'seed', 'batch_size':'total_batch_size', 'num_workers':'num_workers',
                          'nominal_lr':'nominal_lr', 'momentum':'momentum', 'weight_decay':'weight_decay',
                          'critic_lr':'critic_lr', 'critic_betas':'critic_betas', 'detection_weight':'detection_weight',
                          'triplet_weight':'triplet_weight', 'adv_weight':'adversarial_weight', 'initial_weights':'initial_weights'}
        for key, old_key in correspondence.items():
            if baseline[key] != old[old_key]:
                raise ValueError(f'Historical baseline cannot match {key}; include hard_expert in suite')
        for key, expected in [('min_lr_ratio', .01), ('gradient_clip', 2.)]:
            if baseline[key] != expected:
                raise ValueError(f'Historical baseline cannot match {key}; include hard_expert in suite')
        for path_key, hash_key in [('initial_weights', 'initial_sha256'), ('train_txt', 'train_sha256'), ('val_txt', 'val_sha256')]:
            if hashlib.sha256(Path(old[path_key]).read_bytes()).hexdigest() != old[hash_key]:
                raise ValueError(f'Historical input changed: {path_key}')
        if hashlib.sha256(Path(baseline['classes_path']).read_bytes()).hexdigest() != hashlib.sha256((ROOT / 'model_data/rdd_classes.txt').read_bytes()).hexdigest():
            raise ValueError('Historical classes mismatch')
        for expert, path in zip(old['experts'], baseline['expert_paths']):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expert['sha256']:
                raise ValueError('Historical expert hash mismatch')
        historical_eval = ROOT / 'results/test_comparison_standard_conf_20261006_201514/latent_unique'
        evaluation_config = json.loads((historical_eval / 'evaluation_config.json').read_text())
        if hashlib.sha256((historical / 'best_epoch_weights.pth').read_bytes()).hexdigest() != evaluation_config['weights_sha256']:
            raise ValueError('Historical evaluation checkpoint mismatch')
        evaluations['hard_expert'] = str(historical_eval)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output = (args.resume_suite or args.output or ROOT / 'results' / f'latent_suite_{stamp}').resolve()
    if not args.resume_suite:
        output.mkdir(parents=True, exist_ok=False)
    gpu_map = {name: args.gpus[i] if args.gpus else os.environ.get('CUDA_VISIBLE_DEVICES', configs[name]['gpu'])
               for i, name in enumerate(configs)}
    state = dict(status='running', created_at=saved_state['created_at'] if args.resume_suite else stamp,
                 restarted_at=stamp if args.resume_suite else None, experiments=args.experiments,
                 evaluations=evaluations, finished_training=[], current='parallel' if args.gpus else 'sequential',
                 jobs={name: dict(gpu=gpu_map[name], phase='pending') for name in configs},
                 verification=str(args.verification.resolve()), pid=os.getpid(),
                 screen=os.environ.get('STY'))
    if not args.resume_suite:
        (output / 'latent_dg_config.py').write_text((ROOT / 'latent_dg_config.py').read_text())
        for name, config in configs.items():
            write_json(output / f'{name}.json', config)
    else:
        write_json(output / f'status_before_screen_{stamp}.json', saved_state)
    lock = threading.Lock()
    write_json(output / 'status.json', state)
    print('SUITE', output, 'GPU_MAP', gpu_map, 'SCREEN', state['screen'], flush=True)

    def phase(name, value, **extra):
        with lock:
            state['jobs'][name].update(phase=value, **extra)
            write_json(output / 'status.json', state)
        print(name, value, extra, flush=True)

    def execute(name, config):
        for relative, digest in verification['source_sha256'].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Implementation changed during suite: {relative}')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_map[name], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        training = output / 'training' / name
        if not (training / 'completion.json').exists():
            if (training / 'run_config.json').exists():
                if not (training / 'training_state.pt').exists():
                    raise ValueError(f'Incomplete run without epoch checkpoint: {training}')
                command = [sys.executable, '-u', str(ROOT / 'train_latent_ablation.py'), '--resume', str(training)]
                phase(name, 'resuming_training')
            else:
                command = [sys.executable, '-u', str(ROOT / 'train_latent_ablation.py'),
                           '--resolved-config', str(output / f'{name}.json'), '--results-path', str(training)]
                phase(name, 'training')
            with (output / f'{name}_train.log').open('a') as log:
                log.write(f'\nSUITE_LAUNCH {stamp} gpu={gpu_map[name]} command={command!r}\n'); log.flush()
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        completion = json.loads((training / 'completion.json').read_text())
        if completion['smoke_test'] or completion['epochs_completed'] != config['epochs']:
            raise ValueError('Training did not complete requested epoch budget')
        with lock:
            state['finished_training'].append(name)
        phase(name, 'evaluation')
        evaluation = output / 'evaluation' / name
        if not (evaluation / 'completion.json').exists():
            if (evaluation / 'evaluation_config.json').exists():
                raise ValueError(f'Incomplete evaluation must be archived before retry: {evaluation}')
            with (output / f'{name}_test.log').open('a') as log:
                subprocess.run([sys.executable, '-u', str(ROOT / 'get_map_new.py'),
                                '--weights', str(training / 'best_epoch_weights.pth'), '--output', str(evaluation),
                                '--experiment', name], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with lock:
            evaluations[name] = str(evaluation)
        phase(name, 'complete')

    try:
        failures = {}
        with ThreadPoolExecutor(max_workers=len(configs) if args.gpus else 1) as pool:
            futures = {pool.submit(execute, name, config): name for name, config in configs.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures[name] = repr(exc)
                    phase(name, 'failed', error=repr(exc))
        if failures:
            raise RuntimeError(f'Experiment failures (other jobs were allowed to finish): {failures}')
        from utils.summarize_latent_ablation import summarize
        ordered = {name: evaluations[name] for name in settings.EXPERIMENTS if name in evaluations}
        summarize(ordered, output / 'comparison')
        state.update(status='complete', current=None)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(output / 'status.json', state)


if __name__ == '__main__':
    main()
