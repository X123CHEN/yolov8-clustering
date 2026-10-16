"""Summarize four complete standard-confidence seven-domain evaluations."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['01_country_multi_unique', '02_latent_multi_unique', '03_country_kway_unique', '04_latent_kway_unique']
TITLES = ['国家域多专家多判别器', '潜在域多专家多判别器', '国家域单K类判别器', '潜在域单K类判别器']
DOMAINS = ['Japan', 'India', 'United_States', 'Norway', 'new_Czech', 'new_China_Drone', 'new_China_MotorBike']
METRICS = ['coco_map50', 'voc_map50', 'coco_map50_95', 'precision_macro', 'recall_macro', 'f1_macro']


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    if not all((ROOT/r/'completion.json').exists() for r in RUNS):
        print('Waiting for remaining evaluations.')
        return
    domain_rows, class_rows, macro_rows = [], [], []
    for run in RUNS:
        completion = read(ROOT/run/'completion.json')
        assert completion['domains_completed'] == 7 and completion['images'] == 9433
        assert not completion['smoke_test']
        cfg = read(ROOT/run/'evaluation_config.json')
        assert cfg['confidence_mode'] == 'max_class_sigmoid'
        rows = []
        for d in DOMAINS:
            m = read(ROOT/run/d/'metrics.json')
            row = dict(experiment=run, domain=d, images=m['images'],
                       coco_map50=m['coco']['map50'], voc_map50=m['voc']['map50'],
                       coco_map50_95=m['coco']['map50_95'],
                       precision_macro=m['voc']['precision_macro'], recall_macro=m['voc']['recall_macro'],
                       f1_macro=m['voc']['f1_macro'])
            rows.append(row)
            domain_rows.append(row)
            for name, v in m['voc']['per_class'].items():
                c = m['coco']['per_class'][name]
                class_rows.append(dict(experiment=run, domain=d, category=name,
                                       coco_ap50=c['ap50'], voc_ap50=v['ap50'], coco_ap50_95=c['ap50_95'],
                                       precision=v['precision'], recall=v['recall'], f1=v['f1'],
                                       tp=v['true_positive'], fp=v['false_positive'], fn=v['false_negative']))
        for group, domains in [('source', DOMAINS[:3]), ('target', DOMAINS[3:]), ('all', DOMAINS)]:
            selected = [r for r in rows if r['domain'] in domains]
            macro_rows.append(dict(experiment=run, group=group, domains=len(selected),
                                   **{k: sum(r[k] for r in selected)/len(selected) for k in METRICS}))
    write_csv(ROOT/'domain_metrics.csv', domain_rows)
    write_csv(ROOT/'class_metrics.csv', class_rows)
    write_csv(ROOT/'macro_metrics.csv', macro_rows)
    report = ['# 四组标准类别置信度复测结果', '',
              '四组原始最佳权重；seed=42；每组七域9433张图，合计28组测试。置信度为 max(sigmoid(class_logits))。', '',
              '640输入、直接缩放、confidence=0.001、NMS IoU=0.5、无TTA；P/R/F1在score≥0.5下统计。所有表中AP为百分数，平均值按测试域等权计算。', '',
              '| 实验 | 七域 COCO AP50 | 七域 VOC AP50 | 未见四域 COCO AP50 | 未见四域 AP50:95 |',
              '|---|---:|---:|---:|---:|']
    for run, title in zip(RUNS, TITLES):
        a = next(r for r in macro_rows if r['experiment'] == run and r['group'] == 'all')
        t = next(r for r in macro_rows if r['experiment'] == run and r['group'] == 'target')
        report.append(f"| {title} | {a['coco_map50']*100:.4f} | {a['voc_map50']*100:.4f} | {t['coco_map50']*100:.4f} | {t['coco_map50_95']*100:.4f} |")
    for metric, label in [('coco_map50', 'COCO AP50'), ('voc_map50', 'VOC AP50')]:
        report += ['', f'## 各域 {label}（%）', '', '| 实验 | '+' | '.join(DOMAINS)+' |', '|---|'+'---:|'*7]
        for run, title in zip(RUNS, TITLES):
            report.append('| '+title+' | '+' | '.join(f"{next(r for r in domain_rows if r['experiment']==run and r['domain']==d)[metric]*100:.4f}" for d in DOMAINS)+' |')
    report += ['', '## 未见域成对比较', '']
    for left, right in [(2, 0), (3, 1), (1, 0), (3, 2)]:
        a = next(r for r in macro_rows if r['experiment']==RUNS[left] and r['group']=='target')
        b = next(r for r in macro_rows if r['experiment']==RUNS[right] and r['group']=='target')
        report.append(f"- {TITLES[left]} − {TITLES[right]}：COCO AP50 {(a['coco_map50']-b['coco_map50'])*100:+.4f} pp；VOC AP50 {(a['voc_map50']-b['voc_map50'])*100:+.4f} pp。")
    report += ['', '本轮仅seed=42。K-way与多专家方案同时涉及专家、对抗目标及更新方式变化；结果用于完整方案比较。', '',
               'domain_metrics.csv为逐域指标，class_metrics.csv为逐类别指标，macro_metrics.csv含源三域、目标四域、全部七域等权平均。']
    (ROOT/'测试结果报告.md').write_text('\n'.join(report)+'\n')
    (ROOT/'completion.json').write_text(json.dumps(dict(complete=True, evaluations=28, images_per_model=9433), indent=2)+'\n')
    print('SUMMARY_COMPLETE', ROOT)


if __name__ == '__main__':
    main()
