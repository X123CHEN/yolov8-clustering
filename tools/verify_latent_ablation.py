"""GPU smoke and historical Hard-LD migration parity; logs are kept per run."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latent_dg_config as settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/latent_ablation_validation')
    parser.add_argument('--check-only', action='store_true', help='Recheck existing completed smoke artifacts')
    parser.add_argument('--legacy-only', action='store_true', help='Revalidate the legacy hard-expert path after core changes')
    parser.add_argument('--deterministic-backend', action='store_true', help='Pin CUDA algorithm selection in check processes only')
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.check_only:
        output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', settings.COMMON['gpu']),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    commands = [('legacy', [sys.executable, '-u', 'train_style_adv.py', '--experiment', 'latent_unique',
                            '--results_path', str(output / 'legacy'), '--smoke_steps', '2'])]
    commands.append(('legacy_repeat', [sys.executable, '-u', 'train_style_adv.py', '--experiment', 'latent_unique',
                                      '--results_path', str(output / 'legacy_repeat'), '--smoke_steps', '2']))
    names = ['hard_expert'] if args.legacy_only else list(settings.EXPERIMENTS)
    for name in names:
        commands.append((name, [sys.executable, '-u', 'train_latent_ablation.py', '--experiment', name,
                                '--results-path', str(output / name), '--smoke-steps', '2']))
    commands.append(('expert_update_parity', [sys.executable, '-u', 'tools/verify_expert_update.py',
                                             '--output', str(output / 'expert_update_parity.json')]))
    if args.deterministic_backend:
        commands=[(name,[sys.executable,'-u',str(ROOT/'tools/run_deterministic_parity.py')]+command[2:])
                  for name,command in commands]
    for name, command in ([] if args.check_only else commands):
        print('START', name, flush=True)
        with (output / f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect {output / (name + ".log")}')
        print('COMPLETE', name, flush=True)
    import torch
    torch.set_num_threads(2)
    legacy = output / 'legacy/latent_unique'
    migrated = output / 'hard_expert'
    old = json.loads((legacy / 'run_config.json').read_text())
    new = json.loads((migrated / 'run_config.json').read_text())
    keys = ['student_initial_sha256', 'initial_sha256', 'train_order_sha256', 'validation_order_sha256',
            'total_batch_size', 'seed', 'epochs', 'experts', 'discriminators', 'fitted_lr', 'min_lr',
            'detection_weight', 'adversarial_weight', 'triplet_weight']
    for key in keys:
        matches = math.isclose(old[key], new[key], rel_tol=1e-14, abs_tol=0) if isinstance(old[key], float) else old[key] == new[key]
        if not matches:
            raise AssertionError(f'Historical config mismatch: {key}')
    historical = json.loads((ROOT / 'results/comparison_20261005_100ep/latent_unique/run_config.json').read_text())
    for key in keys + ['train_sha256', 'val_sha256']:
        if key == 'validation_order_sha256':
            continue  # Smoke validation is an explicit 13-image prefix.
        matches = math.isclose(historical[key], new[key], rel_tol=1e-14, abs_tol=0) if isinstance(historical[key], float) else historical[key] == new[key]
        if not matches:
            raise AssertionError(f'Completed historical run mismatch: {key}')
    differences = {}
    for filename in ['best_epoch_weights.pth', 'last_epoch_weights.pth']:
        a = torch.load(legacy / filename, map_location='cpu')
        b = torch.load(migrated / filename, map_location='cpu')
        if a.keys() != b.keys():
            raise AssertionError('State-dict key mismatch')
        maximum = 0.
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=2e-5, msg=lambda msg: f'{filename}:{key}: {msg}')
            maximum = max(maximum, float((a[key].double()-b[key].double()).abs().max()))
        differences[filename] = maximum
    old_state = torch.load(legacy / 'training_state.pt', map_location='cpu')
    new_state = torch.load(migrated / 'training_state.pt', map_location='cpu')
    exact = json.loads((output / 'expert_update_parity.json').read_text())
    if not exact['passed'] or exact['max_generator_gradient_difference'] != 0 or exact['max_discriminator_state_difference'] != 0:
        raise AssertionError('Identical-feature expert update parity failed')
    repeated = torch.load(output / 'legacy_repeat/latent_unique/training_state.pt', map_location='cpu')
    def critic_difference(other):
        maximum, outside, total = 0., 0, 0
        for a, b in zip(old_state['discriminators'], other['discriminators']):
            for key in a:
                diff = (a[key].double()-b[key].double()).abs()
                maximum = max(maximum, float(diff.max()))
                outside += int((diff > 2e-6 + 2e-5*a[key].abs()).sum())
                total += diff.numel()
        return dict(max_abs=maximum, outside_strict_tolerance=outside, elements=total)
    critic_noise = critic_difference(repeated)
    migrated_noise = critic_difference(new_state)
    # Adam can amplify near-zero gradient sign changes from cross-process cuDNN
    # algorithm choices. Require exact same-feature parity above, and ensure the
    # end-to-end difference remains within twice the measured legacy repeat noise.
    if (migrated_noise['max_abs'] > 2*critic_noise['max_abs'] + 2e-6 or
        migrated_noise['outside_strict_tolerance'] > 2*critic_noise['outside_strict_tolerance'] + 10):
        raise AssertionError('Migrated critic difference exceeds measured legacy numerical variability')
    for name in names:
        folder = output / name
        completion = json.loads((folder / 'completion.json').read_text())
        config = json.loads((folder / 'run_config.json').read_text())
        audit = json.loads((folder / 'epoch_audit.jsonl').read_text())
        assert completion['smoke_test'] and completion['epochs_completed'] == 1
        assert audit['images_seen'] == 24 and audit['validation_images'] == 13 and audit['experts_frozen']
        assert config['student_initial_sha256'] == old['student_initial_sha256']
        assert config['discriminators'] == (1 if name.endswith('kway') else 2)
        assert len(config['experts']) == (0 if name.endswith('kway') else 2)
        assert config['source_sha256'] == new['source_sha256']
    report = dict(passed=True, steps=2, source_sha256=new['source_sha256'],
                  historical_config_keys=keys, max_weight_difference=differences,
                  exact_expert_update=exact, legacy_repeat_critic_difference=critic_noise,
                  migrated_critic_difference=migrated_noise,
                  student_initial_sha256=new['student_initial_sha256'],
                  historical_run=str(ROOT / 'results/comparison_20261005_100ep/latent_unique'),
                  smoke_runs={name: str(output / name) for name in names},
                  legacy_only=args.legacy_only)
    report['deterministic_check_backend'] = args.deterministic_backend
    report['formal_backend_unchanged'] = True
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
