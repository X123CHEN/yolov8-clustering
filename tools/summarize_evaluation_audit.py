"""Summarize completed re-evaluation evidence without replacing historical logs."""
import argparse
import csv
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['hard_expert', 'soft_expert', 'hard_kway', 'soft_kway']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'results/evaluation_audit_20261007')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    deadline = time.monotonic() + 10800
    while not all((root/name/'audit_complete.json').exists() for name in NAMES):
        failed = [name for name in NAMES if 'Traceback (most recent call last)' in (root/f'{name}.log').read_text(errors='replace')]
        if failed or not args.watch or time.monotonic() > deadline:
            (root/'summary_status.json').write_text(json.dumps(dict(status='needs_review',failed=failed),indent=2)+'\n')
            raise RuntimeError(f'Audit is incomplete or failed: {failed}')
        (root/'summary_status.json').write_text(json.dumps(dict(status='waiting_for_full_test_replays'),indent=2)+'\n')
        time.sleep(10)
    reports = {name:json.loads((root/name/'audit_complete.json').read_text()) for name in NAMES}
    lines = ['# 训练日志与七域测试口径核查报告', '',
             '四组权重均完成同一源域验证集双口径复算，以及全部七域重新推理。原始训练日志和首次测试结果未覆盖。', '',
             '## 源域验证 COCO AP50（%）', '',
             '| 实验 | best 轮次 | 原训练日志 | 统一旧置信度 | 统一标准类别置信度 |',
             '|---|---:|---:|---:|---:|']
    for name,r in reports.items():
        lines.append(f"| {name} | {r['best_epoch']} | {r['logged_map50_at_best']*100:.4f} | {r['modes']['legacy_uncertainty']['coco50']*100:.4f} | {r['modes']['standard_class']['coco50']*100:.4f} |")
    lines += ['', '原四组训练指标均为三源域验证集合并计算的 COCO AP50；差异来自置信度口径。旧 hard_expert 使用不确定性分数，新三组使用标准类别概率，不能直接比较原曲线。',
              '同一 hard_expert 权重只改变置信度，即从约 50.77% 变成 58.61%；这不是模型性能提升。', '',
              '## 全量七域复测核对', '', '| 实验 | 最大 VOC50 差值（百分点） | GT JSON 全部相同 | 预测 JSON 全部相同 |', '|---|---:|---|---|']
    maximum = 0.
    for name,r in reports.items():
        delta=max(abs(d['deltas']['voc_map50']) for d in r['test_replay'])*100
        maximum=max(maximum,delta)
        lines.append(f"| {name} | {delta:.10f} | {all(d['gt_json_equal'] for d in r['test_replay'])} | {all(d['detections_json_equal'] for d in r['test_replay'])} |")
    lines += ['', f'全量复测 VOC50 与首次标准置信度测试的最大绝对差值为 {maximum:.10f} 个百分点。',
              '还使用项目原 VOC 实现对复测预测交叉计分；详情见各组 audit_complete.json 的 independent_voc_delta。', '',
              '## 统一七域 VOC50（%）', '', '| 实验 | Japan | India | US | Norway | Czech | China Drone | China MotorBike | 七域平均 | 源域平均 | 目标域平均 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        with (root/name/'test_replay/summary.csv').open() as f:
            values=[100*float(row['voc_map50']) for row in csv.DictReader(f)]
        lines.append('| '+name+' | '+' | '.join(f'{v:.2f}' for v in values+[sum(values)/7,sum(values[:3])/3,sum(values[3:])/4])+' |')
    lines += ['', '## 一致性约束与限制', '',
              '- 所有 best 权重哈希、测试清单和 decoder 哈希已核对；四组使用最低源域 val loss 选权重，而非最高 map_50。',
              '- 四组源域 train/val 清单一致；每组 100 轮，训练 12,742 张、验证 1,592 张；验证图片 ID 无覆盖，未发现路径层面的 train/val/test 交叉。',
              '- 主训练日志今后固定源域 COCO AP50，默认恢复历史 hard_expert 的置信度口径；禁止 COCO 异常静默回退 VOC。最终测试保持统一标准类别置信度。',
              '- 原每轮训练权重未全部保存，不能凭 best 权重复算整条历史曲线。本表校正的是已保存的 best 权重，不改写原 CSV。',
              '- 原 COCO GT 转换存在 area=w*h-10 的历史写法；本验证集最小框面积为 200，无负面积，不能解释此次 AP50 差异。尺寸分组指标仍应另行规范。',
              '- 七域平均为逐域等权平均；训练日志为三个源域验证集合并后的 COCO AP50，两者不能直接比较。',
              '- 单种子、小幅差值不构成显著性结论；若全量复测差异非零，应先检查逐域明细，不能直接沿用首次结论。']
    (root/'复测核查报告.md').write_text('\n'.join(lines)+'\n')
    (root/'summary_status.json').write_text(json.dumps(dict(status='complete',max_voc50_difference_pp=maximum),indent=2)+'\n')
    print('AUDIT_REPORT_COMPLETE',root/'复测核查报告.md',flush=True)


if __name__=='__main__':main()
