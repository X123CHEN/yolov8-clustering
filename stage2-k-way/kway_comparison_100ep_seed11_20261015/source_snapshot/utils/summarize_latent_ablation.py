"""Four-way source/target macro metrics with strict common evaluation protocol."""
import csv
import json
from pathlib import Path

from utils.summarize_comparison_tests import DOMAINS, METRICS, csv_write


def summarize(evaluations, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configs, rows, classes, macro = {}, [], [], []
    for name, directory in evaluations.items():
        directory = Path(directory)
        completion = json.loads((directory / 'completion.json').read_text())
        if completion['domains_completed'] != 7 or completion['images'] != 9433 or completion['smoke_test']:
            raise ValueError(f'Incomplete formal evaluation: {name}')
        configs[name] = json.loads((directory / 'evaluation_config.json').read_text())
        if configs[name]['confidence_mode'] != 'max_class_sigmoid':
            raise ValueError('Nonstandard confidence mode')
        with (directory / 'summary.csv').open() as f:
            current = list(csv.DictReader(f))
        if [r['domain'] for r in current] != DOMAINS:
            raise ValueError('Domain list mismatch')
        for row in current:
            row['experiment'] = name
            row['images'] = int(row['images'])
            for metric in METRICS:
                row[metric] = float(row[metric])
            values = json.loads((directory / row['domain'] / 'metrics.json').read_text())
            for category, detail in values['voc']['per_class'].items():
                coco = values['coco']['per_class'][category]
                classes.append(dict(experiment=name, domain=row['domain'], category=category,
                                    voc_ap50=detail['ap50'], coco_ap50=coco['ap50'], coco_ap50_95=coco['ap50_95'],
                                    precision=detail['precision'], recall=detail['recall'], f1=detail['f1']))
        rows.extend(current)
        for group, domains in [('source', DOMAINS[:3]), ('target', DOMAINS[3:])]:
            selected = [row for row in current if row['domain'] in domains]
            macro.append(dict(experiment=name, group=group,
                              **{metric: sum(r[metric] for r in selected)/len(selected) for metric in METRICS}))
    protocol_keys = ['confidence', 'nms_iou', 'score_threshold', 'confidence_mode', 'decoder_sha256',
                     'input_shape', 'letterbox', 'tta', 'phi', 'classes', 'test_list_sha256']
    first = next(iter(configs.values()))
    for config in configs.values():
        for key in protocol_keys:
            if config[key] != first[key]:
                raise ValueError(f'Evaluation protocol mismatch: {key}')
    csv_write(output / 'domain_metrics.csv', rows)
    csv_write(output / 'class_metrics.csv', classes)
    csv_write(output / 'macro_metrics.csv', macro)
    deltas = []
    for hard, soft in [('hard_expert', 'soft_expert'), ('hard_kway', 'soft_kway')]:
        for group in ['source', 'target']:
            if hard not in evaluations or soft not in evaluations:
                continue
            a = next(r for r in macro if r['experiment'] == hard and r['group'] == group)
            b = next(r for r in macro if r['experiment'] == soft and r['group'] == group)
            deltas.append(dict(comparison=f'{soft} minus {hard}', group=group,
                               **{m + '_pp': 100*(b[m]-a[m]) for m in METRICS}))
    if deltas:
        csv_write(output / 'comparison_deltas.csv', deltas)
    lines = ['# 固定潜在域四组消融测试报告', '',
             '每组七域 9,433 张；标准类别置信度；源域/目标域分别等权宏平均。权重由源域最低 val loss 选择。', '',
             '| 实验 | 源域 VOC50 (%) | 目标域 VOC50 (%) | 目标域 COCO50:95 (%) |',
             '|---|---:|---:|---:|']
    for name in evaluations:
        s = next(r for r in macro if r['experiment'] == name and r['group'] == 'source')
        t = next(r for r in macro if r['experiment'] == name and r['group'] == 'target')
        lines.append(f"| {name} | {100*s['voc_map50']:.2f} | {100*t['voc_map50']:.2f} | {100*t['coco_map50_95']:.2f} |")
    lines += ['', '单种子结果不代表统计显著性；跨结构比较同时改变专家、判别器结构和对抗目标。', '',
              '各实验原始评估目录（hard_expert 可能来自迁移验证后的历史 latent_unique）：']
    lines += [f'- {name}: `{directory}`' for name, directory in evaluations.items()]
    (output / '测试结果报告.md').write_text('\n'.join(lines) + '\n')
    (output / 'verification.json').write_text(json.dumps(dict(protocol_match=True, evaluations=evaluations), indent=2) + '\n')
