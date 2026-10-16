"""Per-run analysis and the four-way detection/cost comparison."""
import argparse
import csv
import json
from pathlib import Path

EXPERIMENTS = ('01_country_multi_unique', '02_latent_multi_unique',
               '03_country_kway_unique', '04_latent_kway_unique')
TITLES = ('国家域多专家多判别器', '潜在域多专家多判别器', '国家域单K类判别器', '潜在域单K类判别器')
DOMAINS = ('Japan', 'India', 'United_States', 'Norway', 'new_Czech', 'new_China_Drone', 'new_China_MotorBike')
DISPLAY = ('Japan', 'India', 'United States', 'Norway', 'Czech', 'China-Drone', 'China-MotorBike')


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def read_csv(path):
    if not path.exists(): return []
    with path.open() as stream: return list(csv.DictReader(stream))


def number(value):
    return None if value in (None, '') else float(value)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value, scale=1):
    return '待完成' if value is None else f'{value * scale:.3f}'


def write_text(path, content):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(content, encoding='utf-8'); temp.replace(path)


def summarize(suite):
    suite = Path(suite)
    state = read_json(suite / 'status.json')
    rows, metrics = [], {}
    for index, name in enumerate(EXPERIMENTS):
        folder = suite / name
        folder.mkdir(exist_ok=True)
        requested = read_json(suite / 'configs' / f'{name}.json')
        config = read_json(folder / 'run_config.json') or requested
        timing = read_json(folder / 'timing.json')
        checkpoint = read_json(folder / 'best_checkpoint.json')
        training = read_csv(folder / 'metrics/metrics_epoch.csv')
        testing = read_csv(folder / 'testing/summary.csv')
        completion = read_json(folder / 'testing/completion.json')
        test_complete = completion.get('domains_completed') == 7 and not completion.get('smoke_test')
        values = {item['domain']: item for item in testing}
        metrics[name] = values
        row = dict(experiment=name, title=TITLES[index], gpu=state.get('jobs', {}).get(name, {}).get('gpu', index),
                   phase=state.get('jobs', {}).get(name, {}).get('phase', 'pending'),
                   seed=config['seed'], epochs_completed=len(training), selected_epoch=checkpoint.get('epoch'),
                   train_wall_time_s=timing.get('training_s'), test_wall_time_s=timing.get('testing_s'),
                   training_batch_time_s=sum(float(v['train_time_s']) for v in training) if training else None,
                   source_validation_time_s=sum(float(v['val_time_s']) for v in training) if training else None,
                   peak_memory_gb=max(float(v['peak_mem_GB']) for v in training) if training else None,
                   head_parameters=config.get('head_parameters'), expert_parameters=config.get('expert_parameters'),
                   total_training_parameters=config.get('total_training_parameters'))
        steps = sum(int(v['steps']) for v in training)
        row['mean_train_step_s'] = row['training_batch_time_s'] / steps if steps else None
        for algorithm, columns in (('coco', ('map50', 'map50_95')), ('voc', ('map50',))):
            for column in columns:
                for split, domains in (('source', DOMAINS[:3]), ('target', DOMAINS[3:]), ('all', DOMAINS)):
                    key = f'{algorithm}_{column}'
                    row[f'{split}_{key}'] = mean([number(values[d][key]) for d in domains]) if test_complete else None
        rows.append(row)
        lines = [f'# {TITLES[index]}：结果分析', '',
                 f'状态：{row["phase"]}；seed={config["seed"]}；Unique；计划{config["epochs"]}轮，已记录{len(training)}轮。', '',
                 f'最佳权重按最低源验证检测 loss 选择，当前选中 epoch：{checkpoint.get("epoch", "待完成")}。', '',
                 '| 耗时与计算开销 | 数值 |', '|---|---:|',
                 f'| 训练壁钟耗时（秒） | {fmt(row["train_wall_time_s"])} |',
                 f'| 训练 batch 累计耗时（秒） | {fmt(row["training_batch_time_s"])} |',
                 f'| 源验证 loss 累计耗时（秒） | {fmt(row["source_validation_time_s"])} |',
                 f'| 七域测试壁钟耗时（秒） | {fmt(row["test_wall_time_s"])} |',
                 f'| 平均训练步耗时（秒） | {fmt(row["mean_train_step_s"])} |',
                 f'| 峰值显存（GiB） | {fmt(row["peak_memory_gb"])} |',
                 f'| 判别器参数量 | {row["head_parameters"] if row["head_parameters"] is not None else "待初始化"} |',
                 f'| 冻结专家参数量 | {row["expert_parameters"] if row["expert_parameters"] is not None else "待初始化"} |', '',
                 '训练壁钟耗时包含进程初始化、全部训练、源验证、每轮AP与权重保存。测试壁钟耗时包含模型加载、七域推理、VOC/COCO计算和结果保存。', '',
                 '| 测试域 | 图片数 | COCO AP50 (%) | COCO AP50:95 (%) | VOC AP50 (%) | 分域耗时（秒） |',
                 '|---|---:|---:|---:|---:|---:|']
        for domain, display in zip(DOMAINS, DISPLAY):
            item = values.get(domain, {})
            detail = read_json(folder / 'testing' / domain / 'metrics.json')
            lines.append(f'| {display} | {item.get("images", "待完成")} | {fmt(number(item.get("coco_map50")),100)} | '
                         f'{fmt(number(item.get("coco_map50_95")),100)} | {fmt(number(item.get("voc_map50")),100)} | {fmt(detail.get("elapsed_s"))} |')
        lines += ['', f'七域平均 VOC AP50：{fmt(row["all_voc_map50"],100)}%；COCO AP50：{fmt(row["all_coco_map50"],100)}%。', f'四个未见域平均 COCO AP50：{fmt(row["target_coco_map50"],100)}%；AP50:95：{fmt(row["target_coco_map50_95"],100)}%。',
                  f'三个源域平均 COCO AP50：{fmt(row["source_coco_map50"],100)}%。', '']
        if training:
            last = training[-1]
            lines += [f'最后记录轮次的检测/Triplet/域损失：{last["loss_yolo"]} / {last["loss_triplet"]} / {last["loss_generator"]}。', '']
        audit_path = folder / 'epoch_audit.jsonl'
        if audit_path.exists():
            audits = audit_path.read_text().splitlines()
            if audits:
                last = json.loads(audits[-1])
                if 'domain_balanced_accuracy' in last:
                    lines += [f'该轮域均衡准确率：{last["domain_balanced_accuracy"]*100:.2f}%；各域 recall：{last["per_domain_recall"]}。', '']
        if not test_complete:
            lines += ['七域测试完成后分析检测泛化与对应多判别器基线的差值。', '']
        write_text(folder / '结果分析报告.md', '\n'.join(lines))
    with (suite / 'comparison.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (suite / 'comparison.csv.tmp').replace(suite / 'comparison.csv')
    lines = ['# 四组实验对比报告', '', f'四组统一 seed={rows[0]["seed"]}、Unique、{requested["epochs"]}轮及同周期余弦日程、640输入、YOLOv8-s。', '',
             '主指标为七个测试集等权平均 VOC AP50；同时报告七域 COCO AP50、未见四域指标及训练成本。标准类别置信度为 max(sigmoid(class_logits))。所有指标为百分数，差值为百分比点（pp）。', '',
             '| 实验 | GPU | 状态 | 已训练轮数 | 七域VOC AP50 (%) | 七域COCO AP50 (%) | 未见域平均AP50 (%) | 未见域平均AP50:95 (%) | 源域平均AP50 (%) |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["title"]} | {row["gpu"]} | {row["phase"]} | {row["epochs_completed"]} | {fmt(row["all_voc_map50"],100)} | {fmt(row["all_coco_map50"],100)} | {fmt(row["target_coco_map50"],100)} | '
                     f'{fmt(row["target_coco_map50_95"],100)} | {fmt(row["source_coco_map50"],100)} |')
    lines += ['', '| 实验 | 训练耗时（小时） | 七域测试耗时（小时） | 训练步均时（秒） | 峰值显存（GiB） | 总训练模型参数量 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["title"]} | {fmt(row["train_wall_time_s"],1/3600)} | {fmt(row["test_wall_time_s"],1/3600)} | '
                     f'{fmt(row["mean_train_step_s"])} | {fmt(row["peak_memory_gb"])} | {row["total_training_parameters"] if row["total_training_parameters"] is not None else "待初始化"} |')
    lines += ['', '训练耗时包含初始化、源验证、每轮AP与保存；测试耗时包含模型加载、推理、指标计算及保存。总训练模型参数量包含共享Student、判别器及冻结专家。', '',
              '| 测试域 COCO AP50 (%) | 国家多判别器 | 潜在多判别器 | 国家K-way | 潜在K-way |', '|---|---:|---:|---:|---:|']
    for domain, display in zip(DOMAINS, DISPLAY):
        lines.append('| ' + display + ' | ' + ' | '.join(fmt(number(metrics[n].get(domain, {}).get('coco_map50')),100) for n in EXPERIMENTS) + ' |')
    lines += ['', '| 测试域 VOC AP50 (%) | 国家多判别器 | 潜在多判别器 | 国家K-way | 潜在K-way |', '|---|---:|---:|---:|---:|']
    for domain, display in zip(DOMAINS, DISPLAY):
        lines.append('| ' + display + ' | ' + ' | '.join(fmt(number(metrics[n].get(domain, {}).get('voc_map50')),100) for n in EXPERIMENTS) + ' |')
    lines += ['', '## 成对比较', '']
    for left, right, label in ((2,0,'国家单头 vs 国家多判别器'), (3,1,'潜在单头 vs 潜在多判别器'),
                               (1,0,'潜在划分 vs 国家划分（多判别器）'), (3,2,'潜在划分 vs 国家划分（单头）')):
        a, b = rows[left], rows[right]
        if a['all_voc_map50'] is not None and b['all_voc_map50'] is not None:
            delta = 100 * (a['all_voc_map50'] - b['all_voc_map50'])
            lines.append(f'- {label}：七域平均 VOC AP50 差值 {delta:+.3f} pp。')
            if left in (2,3) and right in (0,1):
                speedup = b['train_wall_time_s'] / a['train_wall_time_s']
                lines.append(f'  训练壁钟加速比 {speedup:.3f}×；该 seed 下平均精度{"达到或超过" if delta >= 0 else "低于"}对应基线。')
        else:
            lines.append(f'- {label}：待两组完成七域测试。')
    lines += ['', '## 实验范围', '', '本轮为一个配对seed，不提供多seed均值或显著性结论。四组在同型号GPU上并行运行，训练耗时包含共享CPU与存储负载的影响。',
              'K-way改变了对抗目标和更新方式；两组单头均使用一次源训练batch梯度标定后固定的域系数，具体值及公式见 verification.json 和各组 run_config.json。', '']
    write_text(suite / '四组实验对比报告.md', '\n'.join(lines))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    summarize(parser.parse_args().suite)
