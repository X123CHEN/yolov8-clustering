"""Preflight a changed three-way 60-epoch recipe without target-domain tests.

Run after the parent suite completes and the selected PROFILE is registered in
latent_dg_config.py. Unchanged training code inherits its migration parity proof;
the current configuration gets CPU contract tests and three two-step CUDA runs.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latent_dg_config as settings
from tools.run_latent_refinement import baseline_evidence, NAMES
from utils.summarize_latent_training import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective_config(config):
    result = deepcopy(config)
    for key in ('profile', 'results_root', 'gpu'):
        result.pop(key, None)
    if result['assignment'] == 'hard':
        # Student-t alpha does not change the hard labels or hard G prior.
        result['membership'].pop('alpha', None)
    return result


def run_logged(command, log, env):
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Preflight command failed ({result.returncode}); inspect {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--parent-suite', type=Path, required=True)
    parser.add_argument('--prior-verification', type=Path, required=True)
    parser.add_argument('--gpus', nargs=3, default=['5', '1', '2'])
    parser.add_argument('--revalidate-legacy', action='store_true',
                        help='Run fresh legacy end-to-end and identical-feature parity for changed core code')
    parser.add_argument('--calibration', type=Path, help='Completed source gradient calibration for structural60')
    parser.add_argument('--check-only', action='store_true',
                        help='Recheck completed GPU artifacts against current core hashes without rerunning GPU jobs')
    args = parser.parse_args()
    output, parent = args.output.resolve(), args.parent_suite.resolve()
    if settings.PROFILE != args.profile:
        raise ValueError('Select the requested PROFILE in latent_dg_config.py first')
    if len(set(args.gpus)) != 3:
        raise ValueError('Three distinct GPUs are required')
    parent_audit_path = parent / 'completion_audit.json'
    parent_audit = json.loads(parent_audit_path.read_text())
    if parent_audit['incomplete'] or set(parent_audit['experiments']) != set(NAMES):
        raise ValueError('Wait for and audit all three parent runs before this preflight')
    if any(r['meets_current_target'] for r in parent_audit['experiments'].values()):
        raise ValueError('A parent experiment already meets target; use a subset retry instead of repeating all three')
    for row in parent_audit['experiments'].values():
        if not all(row.get(k) for k in ('complete', 'checkpoint_matches_ema',
                                       'matching_discriminator_state', 'full_60_epoch_budget',
                                       'frozen_inputs_and_sources')):
            raise ValueError('Parent completion evidence is incomplete')
    criteria = json.loads((parent / 'acceptance_criteria.json').read_text())
    if criteria['target_ap50'] < .51 or criteria['selection'] != 'minimum source validation detection loss':
        raise ValueError('Preserve the current >=51% minimum-validation-loss target')
    prior_path = args.prior_verification.resolve()
    prior = json.loads(prior_path.read_text())
    if not prior['passed'] or not prior['expert_update_parity']['passed']:
        raise ValueError('Missing successful training-code migration proof')
    sources = {}
    for relative, digest in prior['source_sha256'].items():
        actual = sha(ROOT / relative)
        if relative != 'latent_dg_config.py' and actual != digest and not args.revalidate_legacy:
            raise ValueError(f'Training or controller code changed: {relative}; revalidate that change before inheriting parity')
        sources[relative] = actual
    configs = {name: settings.resolve(name, args.profile) for name in NAMES}
    calibration = None
    if args.profile == 'structural60':
        if not args.revalidate_legacy or args.calibration is None:
            raise ValueError('Structural changes require fresh legacy parity and recorded gradient calibration')
        calibration = json.loads(args.calibration.read_text())
        if not calibration.get('complete') or calibration.get('reference_is_formal_initialization') is not False:
            raise ValueError('Missing completed two-regime source-only calibration')
        for name,config in configs.items():
            if config['adv_weight'] != calibration['recommended_adv_weight'][name]:
                raise ValueError(f'{name}: formal coefficient differs from frozen calibration rule')
            expected = deepcopy(calibration['measured'][name]['config'])
            expected['adv_weight'] = config['adv_weight']
            if expected != config:
                raise ValueError(f'{name}: configuration changed beyond the calibrated coefficient')
        for relative,digest in calibration['source_sha256'].items():
            if sha(ROOT/relative) != digest:
                raise ValueError(f'Source changed after gradient calibration: {relative}')
    baseline = baseline_evidence()
    for name, config in configs.items():
        previous = json.loads((parent / f'{name}.json').read_text())
        if effective_config(config) == effective_config(previous):
            raise ValueError(f'{name}: no effective change to the failed parent recipe')
    hard, soft = [deepcopy(configs[n]) for n in ('hard_kway', 'soft_kway')]
    for config in (hard, soft):
        config.pop('experiment')
        config.pop('assignment')
    if hard != soft:
        raise ValueError('Paired K-way configurations must differ only by assignment and name')
    if any(c['validation'] != baseline['validation_protocol'] for c in configs.values()):
        raise ValueError('Historical source validation protocol must be preserved')
    sources['latent_dg_config.py'] = sha(ROOT / 'latent_dg_config.py')
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    for relative in ['tools/verify_latent_ablation.py','tools/verify_expert_update.py',
                     'tools/audit_latent_training.py','tools/run_deterministic_parity.py']:
        sources[relative] = sha(ROOT/relative)
    for test in sorted((ROOT/'tests').glob('test_latent_*.py')):
        sources[str(test.relative_to(ROOT))] = sha(test)
    if args.check_only and not output.is_dir():
        raise ValueError('No completed preflight artifacts to recheck')
    output.mkdir(parents=True, exist_ok=args.check_only)
    config_hashes, commands, smoke_configs = {}, [], {}
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    tests = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
             '-p', 'test_latent_*.py', '-v']
    commands.append(dict(stage='cpu_contract_tests', command=tests))
    atomic_json(output / 'commands.json', commands)
    run_logged(tests, output / 'unit_tests.log', env)
    test_text = (output / 'unit_tests.log').read_text()
    count = re.search(r'Ran (\d+) tests?', test_text)
    if not count or int(count.group(1)) < 16 or 'skipped=' in test_text:
        raise ValueError('Expected all source protocol, gradient, routing and selection contracts to run')
    parity = prior['expert_update_parity']
    parity_basis = 'Inherited only for byte-identical training and controller code'
    if args.revalidate_legacy:
        command = [sys.executable, '-u', str(ROOT/'tools/verify_latent_ablation.py'),
                   '--legacy-only', '--deterministic-backend', '--output', str(output/'legacy_revalidation')]
        if args.check_only:command.append('--check-only')
        commands.append(dict(stage='fresh_legacy_migration_parity',command=command,cuda_visible_devices=args.gpus[0]))
        atomic_json(output/'commands.json',commands)
        run_logged(command,output/'legacy_revalidation.log',dict(env,CUDA_VISIBLE_DEVICES=args.gpus[0]))
        fresh_path=output/'legacy_revalidation/verification.json'
        fresh=json.loads(fresh_path.read_text())
        if not fresh['passed'] or not fresh['exact_expert_update']['passed']:
            raise ValueError('Fresh legacy migration proof failed')
        parity=dict(fresh['exact_expert_update'],end_to_end_verification=str(fresh_path),
                    end_to_end_verification_sha256=sha(fresh_path),
                    max_weight_difference=fresh['max_weight_difference'])
        sources.update(fresh['source_sha256'])
        parity_basis='Fresh 2-step legacy repeat/migrated comparison with pinned deterministic check backend, plus exact same-feature G/D parity; formal benchmark recipe unchanged'
    for name, gpu in zip(NAMES, args.gpus):
        frozen = output / f'{name}.json'
        atomic_json(frozen, configs[name])
        config_hashes[name] = sha(frozen)
        folder = output / name
        command = [sys.executable, '-u', str(ROOT / 'train_latent_ablation.py'),
                   '--resolved-config', str(frozen), '--results-path', str(folder),
                   '--smoke-steps', '2']
        commands.append(dict(stage=name, command=command, cuda_visible_devices=gpu))
        atomic_json(output / 'commands.json', commands)
        print(f'SMOKE_START {name} gpu={gpu}', flush=True)
        if not args.check_only:
            run_logged(command, output / f'{name}.log', dict(env, CUDA_VISIBLE_DEVICES=gpu))
        actual = json.loads((folder / 'run_config.json').read_text())
        complete = json.loads((folder / 'completion.json').read_text())
        audits = [json.loads(line) for line in (folder / 'epoch_audit.jsonl').read_text().splitlines()]
        if actual['resolved_config'] != configs[name] or actual['cuda_visible_devices'] != gpu:
            raise ValueError(f'{name}: actual smoke configuration or device differs')
        if not complete['smoke_test'] or complete['epochs_completed'] != 1 or len(audits) != 1:
            raise ValueError(f'{name}: smoke result must remain explicitly non-scientific')
        audit = audits[0]
        if (audit['images_seen'] != 24 or audit['validation_images'] != 13
                or not audit['experts_frozen'] or audit['adversarial_weight_end'] <= 0):
            raise ValueError(f'{name}: invalid two-step training traversal or generator ramp')
        if actual['student_initial_sha256'] != baseline['student_initial_sha256']:
            raise ValueError(f'{name}: detector initialization differs from historical baseline')
        if actual['discriminators'] != (2 if name == 'soft_expert' else 1):
            raise ValueError(f'{name}: discriminator architecture differs')
        if len(actual['experts']) != (2 if name == 'soft_expert' else 0):
            raise ValueError(f'{name}: expert architecture differs')
        for relative, digest in actual['source_sha256'].items():
            if sources.get(relative) != digest:
                raise ValueError(f'{name}: source was not covered by inherited parity: {relative}')
        smoke_configs[name] = actual
        print(f'SMOKE_COMPLETE {name}', flush=True)
    a, b = [smoke_configs[n] for n in ('hard_kway', 'soft_kway')]
    conditional = configs['hard_kway'].get('kway_head') == 'conditional_style'
    actual_prior_key = 'condition_source' if conditional else 'prior'
    if a[actual_prior_key] != b[actual_prior_key] or a['discriminator_initial_sha256'] != b['discriminator_initial_sha256']:
        raise ValueError('K-way generator priors or initial discriminators differ')
    for relative, digest in sources.items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'Source changed during preflight: {relative}')
    report = dict(passed=True, created_at=datetime.now().astimezone().isoformat(),
                  profile=args.profile, unit_tests=int(count.group(1)),
                  unit_test_log_sha256=sha(output / 'unit_tests.log'), source_sha256=sources,
                  config_sha256=config_hashes, baseline=baseline,
                  parent_suite=str(parent), parent_completion_audit_sha256=sha(parent_audit_path),
                  acceptance_criteria=criteria, prior_verification=str(prior_path),
                  prior_verification_sha256=sha(prior_path),
                  expert_update_parity=parity,
                  parity_basis=parity_basis,
                  smoke_runs={n: str(output / n) for n in NAMES},
                  calibration=str(args.calibration.resolve()) if args.calibration else None,
                  calibration_sha256=sha(args.calibration) if args.calibration else None,
                  test_evaluation_performed=False)
    atomic_json(output / 'verification.json', report)
    print(json.dumps(dict(passed=True, profile=args.profile, unit_tests=int(count.group(1)),
                         verification=str(output / 'verification.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
