"""Compare source-validation AP50 only; never invoke target-domain evaluation."""
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def read_jsonl(path):
    if not path.exists(): return []
    rows = []
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass  # An active writer may have an incomplete final line.
    return rows


def metric_rows(path):
    if not path.exists(): return []
    rows = []
    for r in csv.DictReader(path.open()):
        try:
            epoch,val_loss = int(r['epoch']),float(r['val_loss'])
            value = None if r.get('map_50') in ('', None) else float(r['map_50'])
            if (value is not None and (not math.isfinite(value) or not 0<=value<=1)) or not math.isfinite(val_loss):
                raise ValueError('Invalid validation metric')
        except (TypeError,ValueError,KeyError):
            continue
        rows.append(dict(epoch=epoch,map_50=value,val_loss=val_loss))
    return rows


def summarize(suite):
    suite = Path(suite)
    with (suite/'training_summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest = json.loads((suite/'manifest.json').read_text())
        baseline = manifest['baseline']
        rows = []
        for name in manifest['experiments']:
            folder = suite/'training'/name
            metrics = metric_rows(folder/'metrics/metrics_epoch.csv')
            audits = read_jsonl(folder/'epoch_audit.jsonl')
            valid_epochs = {r['epoch'] for r in audits}
            metrics = [r for r in metrics if r['epoch'] in valid_epochs]
            if len({r['epoch'] for r in metrics})!=len(metrics): raise ValueError('Duplicate committed epoch')
            evaluated = [r for r in metrics if r['map_50'] is not None]
            config_path = folder/'run_config.json'
            period = json.loads(config_path.read_text()).get('coco_period', 1) if config_path.exists() else 1
            if metrics:
                protocol = json.loads((folder/'validation_protocol.json').read_text())
                if any(protocol.get(k)!=v for k,v in baseline['validation_protocol'].items()):
                    raise ValueError(f'{name}: incomparable validation protocol')
                coco = {r['epoch']:r for r in read_jsonl(folder/'metrics/validation_coco_metrics.jsonl')
                        if r.get('purpose', 'scheduled') == 'scheduled'}
                for m in evaluated:
                    c = coco[m['epoch']]
                    if c['images']!=1592 or c['metric']!='source_val_coco_ap50' or c['confidence_mode']!='legacy_uncertainty':
                        raise ValueError(f'{name}: invalid source metric identity')
                    if abs(c['value']-m['map_50'])>1e-12: raise ValueError('CSV/COCO evidence mismatch')
                if any(r['images_seen']!=12742 or r['validation_images']!=1592 for r in audits):
                    raise ValueError('Dataset traversal changed')
            peak = max(evaluated,key=lambda r:r['map_50']) if evaluated else None
            selected = min(metrics,key=lambda r:r['val_loss']) if metrics else None
            selected = dict(selected) if selected else None
            complete_path = folder/'completion.json'
            complete = False
            if complete_path.exists():
                completion = json.loads(complete_path.read_text())
                complete = (not completion['smoke_test'] and completion['epochs_completed']==60
                            and [r['epoch'] for r in metrics]==list(range(1,61)))
                if not complete: raise ValueError(f'{name}: incomplete formal budget')
                if [r['epoch'] for r in evaluated] != list(range(period, 61, period)):
                    raise ValueError('Scheduled COCO measurements do not match the frozen cadence')
                best = json.loads((folder/'best_map_checkpoint.json').read_text())
                if best['epoch']!=peak['epoch'] or abs(best['map_50']-peak['map_50'])>1e-12:
                    raise ValueError('Best AP50 checkpoint does not match curve')
                best_val = json.loads((folder/'best_checkpoint.json').read_text())
                if best_val['epoch']!=selected['epoch'] or abs(best_val['val_loss']-selected['val_loss'])>1e-12:
                    raise ValueError('Selected checkpoint must minimize validation detection loss')
                weights = folder/'best_epoch_weights.pth'
                if hashlib.sha256(weights.read_bytes()).hexdigest()!=best_val['weights_sha256']:
                    raise ValueError('Selected detector weights do not match checkpoint provenance')
                if selected['map_50'] is None:
                    replay = json.loads((folder/'selected_checkpoint_coco.json').read_text())
                    proof = [r for r in read_jsonl(folder/'metrics/validation_coco_metrics.jsonl')
                             if r.get('purpose') == 'selected_checkpoint_replay']
                    if (replay['epoch'] != selected['epoch'] or replay['weights_sha256'] != best_val['weights_sha256']
                        or replay['validation_images'] != 1592 or replay['metric'] != 'source_val_coco_ap50'
                        or replay['confidence_mode'] != 'legacy_uncertainty' or len(proof) != 1
                        or proof[0]['epoch'] != selected['epoch'] or proof[0]['value'] != replay['map_50']
                        or proof[0]['weights_sha256'] != replay['weights_sha256']
                        or not math.isfinite(replay['map_50']) or not 0 <= replay['map_50'] <= 1):
                        raise ValueError('Invalid minimum-loss checkpoint COCO replay')
                    selected['map_50'] = replay['map_50']
                if best_val.get('map_50')!=selected['map_50']:
                    raise ValueError('Selected checkpoint AP50 does not match its epoch')
                if not (folder/'best_val_training_state.pt').is_file():
                    raise ValueError('Missing matching student/discriminator state')
            rows.append(dict(experiment=name,epochs_completed=len(metrics),complete=complete,
                selected_map_50=selected['map_50'] if selected else None,
                selected_epoch=selected['epoch'] if selected else None,
                selected_val_loss=selected['val_loss'] if selected else None,
                peak_map_50=peak['map_50'] if peak else None,peak_epoch=peak['epoch'] if peak else None,
                last_map_50=evaluated[-1]['map_50'] if evaluated else None,
                delta_pp=(selected['map_50']-baseline['selected_map_50'])*100 if selected and selected['map_50'] is not None else None,
                exceeds_baseline=bool(selected and selected['map_50'] is not None and selected['map_50']>baseline['selected_map_50']),
                reaches_51=bool(complete and selected and selected['map_50'] >= .51),
                training_path=str(folder)))
        all_complete = all(r['complete'] for r in rows)
        result = dict(baseline=baseline,experiments=rows,all_training_complete=all_complete,
            all_three_exceed_baseline=all_complete and all(r['exceeds_baseline'] for r in rows),
            all_three_reach_51=all_complete and all(r['reaches_51'] for r in rows),
            test_evaluation_performed=False,comparison='source validation COCO AP50; historical confidence')
        atomic_json(suite/'training_comparison.json',result)
        with (suite/'training_comparison.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        lines = ['# 60 轮训练源域验证对比','',
            '只比较源域验证 COCO AP50，保持历史 hard_expert 的 legacy_uncertainty、conf=0.05、NMS=0.5、max_boxes=100；未运行七域测试。',
            '',f"历史 hard_expert：最低 val loss 选出的第 **{baseline['selected_epoch']}** 轮，COCO AP50 **{baseline['selected_map_50']*100:.4f}%**，val loss={baseline['selected_val_loss']:.8f}。",
            '', '三组均先按最低 detection val loss 选择权重，再比较该轮 AP50；曲线峰值只作辅助，不作为通过门槛。',
            '', '| 实验 | 已完成轮数 | 最低 val loss 对应 AP50 (%) | 所选轮次 | 与基线差值 (百分点) | 超过基线 | 辅助：曲线峰值 (%) |',
            '|---|---:|---:|---:|---:|---|---:|']
        for r in rows:
            peak = f"{r['peak_map_50']*100:.4f}" if r['peak_map_50'] is not None else '待完成验证'
            selected_value = f"{r['selected_map_50']*100:.4f}" if r['selected_map_50'] is not None else '待完成验证'
            delta = f"{r['delta_pp']:+.4f}" if r['delta_pp'] is not None else '—'
            lines.append(f"| {r['experiment']} | {r['epochs_completed']}/60 | {selected_value} | {r['selected_epoch'] or '—'} | {delta} | {'是' if r['exceeds_baseline'] else '否'} | {peak} |")
        lines.extend(['',f"三组完整 60 轮均超过基线：**{'是' if result['all_three_exceed_baseline'] else '尚未达到'}**。",
            '', f"三组完整 60 轮、最低 val loss 所选权重均达到 51%：**{'是' if result['all_three_reach_51'] else '尚未达到'}**。",
            '', 'AP50 未测轮次留空；先从全部逐轮 val loss 中选权重，最终必要时补测该权重。辅助峰值仅指按既定频率实际评估的节点。',
            '', '该门槛只代表源域验证结果。多参数调整后的结果属于改进方案比较；不能归因为单个因素，也不能作为目标域测试提升的证明。',
            '', '`best_epoch_weights.pth` 继续按最低 detection val loss 保存；`best_map_epoch_weights.pth` 按最高源域 AP50 保存，二者分别记录，避免混淆。'])
        tmp = suite/'训练对比报告.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(suite/'训练对比报告.md')
        return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('suite',type=Path)
    args=parser.parse_args();print(json.dumps(summarize(args.suite),ensure_ascii=False,indent=2))
