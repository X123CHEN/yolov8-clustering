"""Audit the actual training deliverables; does not run inference or training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.summarize_latent_training import atomic_json, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(suite):
    manifest = json.loads((suite/'manifest.json').read_text())
    criteria_path = suite/'acceptance_criteria.json'
    criteria = json.loads(criteria_path.read_text())
    target = criteria['target_ap50']
    assert 0.51 <= target <= 1, 'Current target must be at least 51% AP50'
    assert criteria['comparison'] == '>='
    assert criteria['epochs_per_attempt'] == 60
    assert criteria['metric'] == 'source_val_coco_ap50'
    assert criteria['confidence_mode'] == 'legacy_uncertainty'
    assert criteria['selection'] == 'minimum source validation detection loss'
    assert criteria['test_evaluation'] is False
    assert set(manifest['experiments']) == set(criteria['experiments']) == {'soft_expert', 'hard_kway', 'soft_kway'}
    comparison = summarize(suite)
    baseline = manifest['baseline']
    old_folder = Path(baseline['training_path'])
    old_best = json.loads((old_folder/'best_checkpoint.json').read_text())
    old_rows = list(csv.DictReader((old_folder/'metrics/metrics_epoch.csv').open()))
    old_selected = min(old_rows, key=lambda r: float(r['val_loss']))
    old_config = json.loads((old_folder/'run_config.json').read_text())
    assert old_best['epoch'] == int(old_selected['epoch']) == baseline['selected_epoch'] == 31
    assert float(old_selected['map_50']) == baseline['selected_map_50']
    assert sha(old_folder/'best_epoch_weights.pth') == baseline['selected_weights_sha256']
    assert sha(old_folder/'metrics/metrics_epoch.csv') == baseline['metrics_sha256']
    assert sha(old_folder/'run_config.json') == baseline['config_sha256']
    assert not (suite/'evaluation').exists(), 'Unexpected target evaluation directory'
    assert not list(suite.glob('*_test.log')), 'Unexpected target evaluation logs'
    assert baseline['selected_map_50'] == criteria['baseline_ap50']
    result = dict(achieved=False, target_map_50=target, target_comparison='>=',
                  acceptance_criteria_sha256=sha(criteria_path),
                  baseline_map_50=baseline['selected_map_50'], baseline_epoch=31,
                  selection='minimum source validation detection loss',
                  inference_performed=False, experiments={}, incomplete=[])
    for row in comparison['experiments']:
        name = row['experiment']
        folder = suite/'training'/name
        checks = dict(complete=row['complete'], exceeds_baseline=row['exceeds_baseline'],
                      selected_epoch=row['selected_epoch'], selected_map_50=row['selected_map_50'])
        checks['selected_metric_reaches_target'] = (row['selected_map_50'] is not None
                                                   and row['selected_map_50'] >= target)
        checks['meets_current_target'] = bool(row['complete'] and checks['selected_metric_reaches_target']
                                             and row['exceeds_baseline'])
        result['experiments'][name] = checks
        frozen = json.loads((suite/f'{name}.json').read_text())
        assert sha(suite/f'{name}.json') == manifest['config_sha256'][name]
        if not (folder/'run_config.json').exists():
            result['incomplete'].append(name)
            continue
        cfg = json.loads((folder/'run_config.json').read_text())
        assert cfg['resolved_config'] == frozen
        assert cfg['epochs'] == 60 and cfg['scheduler_epochs'] == 100
        assert cfg['cuda_visible_devices'] == manifest['gpus'][name]
        assert cfg['student_initial_sha256'] == baseline['student_initial_sha256']
        for key in ['train_sha256', 'val_sha256', 'train_order_sha256', 'validation_order_sha256', 'initial_sha256']:
            assert cfg[key] == old_config[key], (name, key)
        for relative, digest in cfg['source_sha256'].items():
            assert sha(folder/'source_snapshot'/relative) == digest, (name, relative)
        assert frozen['assignment'] == ('soft' if name.startswith('soft') else 'hard')
        expected_d = 2 if name == 'soft_expert' else 1
        assert cfg['discriminators'] == expected_d
        assert len(cfg['experts']) == (2 if name == 'soft_expert' else 0)
        if not row['complete']:
            result['incomplete'].append(name)
            continue
        import torch
        torch.set_num_threads(2)
        best = json.loads((folder/'best_checkpoint.json').read_text())
        state = torch.load(folder/'best_val_training_state.pt', map_location='cpu')
        weights = torch.load(folder/'best_epoch_weights.pth', map_location='cpu')
        assert state['epoch'] == best['epoch'] == row['selected_epoch']
        assert state['best_val_loss'] == best['val_loss'] == row['selected_val_loss']
        assert len(state['discriminators']) == expected_d
        assert weights.keys() == state['ema'].keys()
        assert all(torch.equal(value, state['ema'][key]) for key, value in weights.items())
        assert sha(folder/'best_epoch_weights.pth') == best['weights_sha256']
        last = torch.load(folder/'training_state.pt', map_location='cpu')
        assert last['epoch'] == 60 and last['config']['smoke_steps'] == 0
        audits = [json.loads(line) for line in (folder/'epoch_audit.jsonl').read_text().splitlines()]
        assert [a['epoch'] for a in audits] == list(range(1, 61))
        assert all(a['images_seen'] == 12742 and a['validation_images'] == 1592
                   and a['domain_images'] == [5913, 6829] and a['experts_frozen'] for a in audits)
        if frozen.get('profile') == 'structural60':
            assert cfg['coco_period'] == 5 and cfg['loss_validation_period'] == 1
            assert cfg['condition_source']['images'] == 12742
            if name == 'soft_expert':
                assert cfg['expert_normalization'] == 'group'
                assert cfg['expert_reference_routing'] == 'sample_soft'
                assert all(sum(a['sampled_reference_counts']) == 12742 for a in audits)
                assert state.get('routing_rng') is not None and last.get('routing_rng') is not None
                assert all('running_mean' not in k and 'running_var' not in k
                           for d in state['discriminators'] for k in d)
            else:
                assert frozen['kway_head'] == 'conditional_style'
                assert cfg['adversarial_feature'] == 'feat1_style'
                assert cfg['kway_confusion_target'] == 'conditional_hard_prior'
        checks.update(checkpoint_matches_ema=True, matching_discriminator_state=True,
                      full_60_epoch_budget=True, frozen_inputs_and_sources=True)
        del state, weights, last
    kway = [json.loads((suite/'training'/n/'run_config.json').read_text())
            for n in ['hard_kway','soft_kway'] if (suite/'training'/n/'run_config.json').exists()]
    if len(kway) == 2:
        if kway[0]['resolved_config'].get('kway_head') != 'conditional_style':
            assert kway[0]['prior'] == kway[1]['prior']
        assert kway[0]['discriminator_initial_sha256'] == kway[1]['discriminator_initial_sha256']
        assert kway[0].get('condition_source') == kway[1].get('condition_source')
    result['achieved'] = (not result['incomplete'] and len(result['experiments']) == 3
                          and all(r['meets_current_target'] for r in result['experiments'].values()))
    atomic_json(suite/'completion_audit.json', result)
    lines = ['# 当前目标与达标状态', '',
             f'当前门槛：三组各完成 60 轮，最低源域 detection val loss 对应权重的 COCO AP50 均至少 {target*100:.4f}%。',
             f"历史 hard_expert 第 31 轮基准为 {baseline['selected_map_50']*100:.4f}%；仅超过历史基准不等于达到当前门槛。", '',
             '| 实验 | 已完成轮数 | 所选轮次 | 对应 AP50 (%) | 完成 60 轮且达到当前门槛 |',
             '|---|---:|---:|---:|---|']
    for row in comparison['experiments']:
        value = f"{row['selected_map_50']*100:.4f}" if row['selected_map_50'] is not None else '待验证'
        passed = result['experiments'][row['experiment']]['meets_current_target']
        lines.append(f"| {row['experiment']} | {row['epochs_completed']}/60 | {row['selected_epoch'] or '—'} | {value} | {'是' if passed else '否'} |")
    lines.extend(['', f"三组当前目标均达到：**{'是' if result['achieved'] else '尚未达到'}**。", '',
                  '未达标的方案完成本次训练后继续调整，每次保留独立配置、完整日志和权重。曲线峰值只作辅助；暂不运行七域测试。',
                  '本文件由 tools/audit_latent_training.py 生成，状态以生成时已提交的 epoch 为准；源域验证结果不能证明目标域趋势一致。'])
    report = suite/'当前目标与达标状态.md'
    temporary = report.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(report)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--require-achieved', action='store_true')
    args = parser.parse_args()
    result = audit(args.suite.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_achieved and not result['achieved']:
        raise SystemExit(2)
