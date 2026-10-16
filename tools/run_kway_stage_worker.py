"""One v2 experiment per screen; immediately test its minimum-val-loss weights."""
import argparse
import csv
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
NAMES = ('hard_expert', 'hard_kway', 'hard_kway_grl')
DOMAINS = ('Japan', 'India', 'United_States', 'Norway', 'new_Czech',
           'new_China_Drone', 'new_China_MotorBike')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation_rows(directory):
    with (directory / 'summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if [r['domain'] for r in rows] != list(DOMAINS) or sum(int(r['images']) for r in rows) != 9433:
        raise ValueError('Seven-domain evaluation is incomplete or has different sample counts')
    values = [float(r['voc_map50']) for r in rows]
    return rows, dict(zip(DOMAINS, values)), sum(values) / 7


def report(suite, status):
    """Only compare completed tests; source AP is a record, never a decision gate."""
    baseline = Path(status['historical_evaluation'])
    _, historical_values, historical_mean = evaluation_rows(baseline)
    columns = dict(historical_hard_expert=historical_values)
    means = dict(historical_hard_expert=historical_mean)
    training = {}
    for name in NAMES:
        if status['jobs'][name]['phase'] != 'complete':
            continue
        _, columns[name], means[name] = evaluation_rows(suite / 'evaluation' / name)
        directory = suite / 'training' / name
        selected = read(directory / 'best_checkpoint.json')
        peak = read(directory / 'best_map_checkpoint.json')
        training[name] = dict(min_val_epoch=selected['epoch'], min_val_loss=selected['val_loss'],
            min_val_source_coco_ap50=selected['map_50'], peak_observed_source_coco_ap50=peak['map_50'],
            peak_observed_source_epoch=peak['epoch'])
    folder = suite / 'comparison'
    folder.mkdir(exist_ok=True)
    with (folder / 'domain_metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['domain'] + list(columns))
        for domain in DOMAINS:
            writer.writerow([domain] + [values[domain] for values in columns.values()])
    with (folder / 'macro_metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['experiment', 'source_mean', 'target_mean', 'seven_domain_mean'])
        for name, values in columns.items():
            writer.writerow([name, sum(values[d] for d in DOMAINS[:3]) / 3,
                             sum(values[d] for d in DOMAINS[3:]) / 4, means[name]])
    deltas = {}
    for name in NAMES:
        if name in means:
            deltas[name + '_minus_historical'] = means[name] - historical_mean
    for left, right in [('hard_kway', 'hard_expert'), ('hard_kway_grl', 'hard_expert'),
                        ('hard_kway_grl', 'hard_kway')]:
        if left in means and right in means:
            deltas[left + '_minus_' + right] = means[left] - means[right]
    complete = all(status['jobs'][name]['phase'] == 'complete' for name in NAMES)
    decision = dict(metric='equal-weight seven-domain VOC AP50', source_ap_used_as_gate=False,
                    historical_threshold=historical_mean, stage1_complete=complete,
                    effective_candidates=[], selected_route=None)
    if complete:
        candidates = [name for name in NAMES[1:] if means[name] >= historical_mean]
        decision['effective_candidates'] = candidates
        if candidates:
            selected = max(candidates, key=lambda name: means[name])
            decision.update(selected_route=selected, next_stage='fixed soft K-way with the selected objective',
                            objective=read(suite / (selected + '.json'))['adversarial_objective'])
        else:
            decision['next_stage'] = 'Analyze K-way before advancing; no automatic parameter sweep'
    write(folder / 'decision.json', decision)
    write(folder / 'training_metrics.json', training)
    write(folder / 'comparison_deltas.json', deltas)
    lines = ['# 第一阶段 v2 七域测试报告', '',
        '测试权重：最低源域检测验证 loss；全部正式训练60轮。源域COCO AP50保留训练记录，不参与有效性判断。', '',
        '当前报告' + ('已完成三组测试。' if complete else '仅包含已完成的测试，未完成组不参与比较。'), '',
        '| 域 | ' + ' | '.join(columns) + ' |',
        '|---|' + '---:|' * len(columns)]
    for domain in DOMAINS:
        lines.append('| ' + domain + ' | ' + ' | '.join(f'{v[domain]*100:.4f}' for v in columns.values()) + ' |')
    lines.append('| 七域均值 | ' + ' | '.join(f'{means[n]*100:.4f}' for n in columns) + ' |')
    lines += ['', '以上数值单位为百分比，七域等权，不按测试图片数量加权。', '',
              '## 对照差值（百分点）', '']
    lines += [f'- {name}: {value*100:+.4f}' for name, value in deltas.items()]
    lines += ['', '## 源域训练记录（辅助）', '',
              '| 组 | 最低loss轮次 | 对应COCO AP50 | 最高已测COCO AP50 | 最高已测轮次 |',
              '|---|---:|---:|---:|---:|']
    for name, v in training.items():
        lines.append(f"| {name} | {v['min_val_epoch']} | {v['min_val_source_coco_ap50']*100:.4f} | "
                     f"{v['peak_observed_source_coco_ap50']*100:.4f} | {v['peak_observed_source_epoch']} |")
    lines += ['', '源域AP按每5轮评估，最高值指已测轮次的最高值；最低loss轮次若不在周期上则训练后补测。', '',
              '## 继续判断', '', '历史expert七域基准：' + f'{historical_mean*100:.4f}%。']
    if complete:
        lines += ['所选路线：' + str(decision['selected_route']) + '。',
                  '下一步：' + decision['next_stage'] + '。']
        if 'hard_expert' in means:
            for name in decision['effective_candidates']:
                if means[name] < means['hard_expert']:
                    lines.append(f'{name}达到历史基准，但低于同配方expert对照；该差异必须在机制分析中保留。')
    (folder / '测试结果报告.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--experiment', choices=NAMES, required=True)
    args = parser.parse_args()
    suite, name = args.suite.resolve(), args.experiment
    lock = (suite / (name + '.worker.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(suite / (name + '.json'))
    if config['epochs'] != 60 or not config['source_ap_enabled'] or config['coco_period'] != 5:
        raise ValueError('Requested 60-epoch training and source AP recording must be preserved')
    for relative, sha in read(suite / 'implementation.json')['source_sha256'].items():
        if digest(ROOT / relative) != sha:
            raise ValueError('Implementation changed before launch: ' + relative)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=config['gpu'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    training, evaluation = suite / 'training' / name, suite / 'evaluation' / name

    def update(phase, **extra):
        with (suite / 'status.lock').open('a') as shared:
            fcntl.flock(shared, fcntl.LOCK_EX)
            status = read(suite / 'status.json')
            status['jobs'][name].update(phase=phase, pid=os.getpid(), screen=os.environ.get('STY'),
                updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **extra)
            if all(j['phase'] == 'complete' for j in status['jobs'].values()):
                status['status'] = 'complete'
            elif any(j['phase'] == 'failed' for j in status['jobs'].values()):
                status['status'] = 'running_with_failures'
            else:
                status['status'] = 'running'
            write(suite / 'status.json', status)
            if phase == 'complete':
                report(suite, status)
        print(name, phase, extra, flush=True)

    def run(command, filename):
        with (suite / filename).open('ab', buffering=0) as stream:
            stream.write(('\nLAUNCH ' + repr(command) + '\n').encode())
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        update('training', gpu=config['gpu'])
        run([sys.executable, '-u', str(ROOT / 'train_latent_ablation.py'), '--resolved-config',
             str(suite / (name + '.json')), '--results-path', str(training)], name + '_train.log')
        completed = read(training / 'completion.json')
        if completed['smoke_test'] or completed['epochs_completed'] != 60:
            raise ValueError('Training did not complete its formal budget')
        selected = read(training / 'best_checkpoint.json')
        weights = training / 'best_epoch_weights.pth'
        if digest(weights) != selected['weights_sha256']:
            raise ValueError('Selected weights do not match the minimum-loss checkpoint')
        with (training / 'metrics' / 'metrics_epoch.csv').open() as stream:
            epochs = list(csv.DictReader(stream))
        minimum = min(epochs, key=lambda row: float(row['val_loss']))
        if len(epochs) != 60 or int(minimum['epoch']) != selected['epoch']:
            raise ValueError('Minimum validation-loss epoch does not match selected weights')
        update('evaluation', selected_epoch=selected['epoch'], weights_sha256=selected['weights_sha256'])
        run([sys.executable, '-u', str(ROOT / 'get_map_new.py'), '--weights', str(weights),
             '--output', str(evaluation), '--experiment', name], name + '_test.log')
        completed = read(evaluation / 'completion.json')
        if completed['smoke_test'] or completed['domains_completed'] != 7 or completed['images'] != 9433:
            raise ValueError('Test did not complete all seven domains')
        test_config = read(evaluation / 'evaluation_config.json')
        if test_config['weights_sha256'] != selected['weights_sha256']:
            raise ValueError('Test weights differ from the selected training checkpoint')
        historical = read(Path(read(suite / 'status.json')['historical_evaluation']) / 'evaluation_config.json')
        for key in ('confidence', 'nms_iou', 'confidence_mode', 'decoder_sha256', 'input_shape',
                    'letterbox', 'tta', 'strict_load', 'classes', 'test_list_sha256'):
            if test_config[key] != historical[key]:
                raise ValueError('Historical test protocol differs: ' + key)
        _, _, mean = evaluation_rows(evaluation)
        update('complete', seven_domain_mean=mean)
    except BaseException as error:
        update('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
