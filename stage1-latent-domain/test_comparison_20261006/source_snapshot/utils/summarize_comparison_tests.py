"""Summarize only complete seven-domain evaluations, with equal domain weights."""
import argparse
import csv
import json
from pathlib import Path

RUNS = ['baseline','country_repeat','latent_repeat','country_unique','latent_unique']
DOMAINS = ['Japan','India','United_States','Norway','new_Czech','new_China_Drone','new_China_MotorBike']
METRICS = ['voc_map50','coco_map50','coco_map50_95','precision_macro','recall_macro','f1_macro']


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root;all_rows=[];classes=[];macro=[];configs={}
    for name in RUNS:
        completion=json.loads((root/name/'completion.json').read_text())
        assert completion['domains_completed']==7 and completion['images']==9433 and not completion['smoke_test']
        configs[name]=json.loads((root/name/'evaluation_config.json').read_text())
        rows=list(csv.DictReader((root/name/'summary.csv').open()))
        assert [row['domain'] for row in rows]==DOMAINS
        for row in rows:
            for metric in METRICS:row[metric]=float(row[metric])
            row['images']=int(row['images']);all_rows.append(row)
            m=json.loads((root/name/row['domain']/'metrics.json').read_text())
            assert len(list((root/name/row['domain']/'detection-results').glob('*.txt')))==row['images']
            assert len(list((root/name/row['domain']/'ground-truth').glob('*.txt')))==row['images']
            for category,values in m['voc']['per_class'].items():
                classes.append(dict(experiment=name,domain=row['domain'],category=category,
                                    voc_ap50=values['ap50'],coco_ap50=m['coco']['per_class'][category]['ap50'],
                                    coco_ap50_95=m['coco']['per_class'][category]['ap50_95'],
                                    precision=values['precision'],recall=values['recall'],f1=values['f1'],
                                    gt_objects=values['gt_objects'],tp=values['true_positive'],fp=values['false_positive'],fn=values['false_negative']))
        for group,domains in [('source',DOMAINS[:3]),('target',DOMAINS[3:]),('all',DOMAINS)]:
            selected=[row for row in rows if row['domain'] in domains]
            macro.append(dict(experiment=name,group=group,domains=len(selected),
                              **{metric:sum(r[metric] for r in selected)/len(selected) for metric in METRICS}))
    for key in ['confidence','nms_iou','score_threshold','input_shape','letterbox','tta','phi','classes','test_list_sha256']:
        assert all(c[key]==configs['baseline'][key] for c in configs.values()), key
    deltas=[]
    for left,right in [('latent_repeat','country_repeat'),('latent_unique','country_unique'),('country_repeat','baseline'),('latent_repeat','baseline'),('country_unique','country_repeat'),('latent_unique','latent_repeat')]:
        for domain in DOMAINS+['source_macro','target_macro','all_macro']:
            pool=macro if domain.endswith('_macro') else all_rows
            field='group' if domain.endswith('_macro') else 'domain'; value=domain.removesuffix('_macro')
            a=next(r for r in pool if r['experiment']==left and r[field]==value)
            b=next(r for r in pool if r['experiment']==right and r[field]==value)
            deltas.append(dict(comparison=left+' minus '+right,domain=domain,
                               **{metric+'_pp':100*(a[metric]-b[metric]) for metric in METRICS}))
    csv_write(root/'domain_metrics.csv',all_rows);csv_write(root/'class_metrics.csv',classes)
    csv_write(root/'macro_metrics.csv',macro);csv_write(root/'comparison_deltas.csv',deltas)
    report=['# 五模型七域测试报告','',
            '35/35 组评估完成；每个模型测试 9,433 张图片，共 47,165 次模型-图片推理。所有权重均由源域验证集最低 val loss 选择，未根据测试集选权重。','',
            '统一设置：YOLOv8-s，640×640，直接缩放，confidence=0.001，NMS IoU=0.5，无TTA。P/R/F1固定在score≥0.5、IoU≥0.5；各类别P/R/F1取宏平均。','',
            'VOC mAP@50使用连续PR曲线积分与像素包含式IoU；COCO mAP使用pycocotools、101点插值、连续坐标、maxDets=100。两种AP口径分别报告，不能直接混用。预测坐标与置信度保留精度，COCO面积为宽×高。','',
            '## 各域 VOC mAP@50（%）','',
            '| 实验 | '+' | '.join(DOMAINS)+' |','|---|'+'---:|'*7]
    for name in RUNS:
        report.append('| '+name+' | '+' | '.join(f"{next(r for r in all_rows if r['experiment']==name and r['domain']==d)['voc_map50']*100:.2f}" for d in DOMAINS)+' |')
    report+=['','## 源域与目标域宏平均（%）','',
             'India/Japan/United_States为源域测试；其余四域为未见目标域。每个域等权，不按图片数加权，也不是把所有图片合并计算一次AP。','',
             '| 实验 | 源域 VOC50 | 目标域 VOC50 | 目标域 COCO50 | 目标域 COCO50:95 |','|---|---:|---:|---:|---:|']
    for name in RUNS:
        s=next(r for r in macro if r['experiment']==name and r['group']=='source')
        t=next(r for r in macro if r['experiment']==name and r['group']=='target')
        report.append(f"| {name} | {s['voc_map50']*100:.2f} | {t['voc_map50']*100:.2f} | {t['coco_map50']*100:.2f} | {t['coco_map50_95']*100:.2f} |")
    report+=['','## 核心对比','']
    for name in ['latent_repeat minus country_repeat','latent_unique minus country_unique']:
        d=next(r for r in deltas if r['comparison']==name and r['domain']=='target_macro')
        report.append(f"- {name}：四目标域 VOC mAP@50 差值 {d['voc_map50_pp']:+.2f} 个百分点；COCO mAP@50:95 差值 {d['coco_map50_95_pp']:+.2f} 个百分点。")
    report+=['','去重组的图片与更新预算一致，适合比较域划分；不去重组同时包含采样比例和更新次数差异。全部实验仅一个随机种子，测试结果不能单独证明统计显著性。Cluster_1专家按用户确认使用中断训练前的第52轮best。','',
             '逐类别结果见 class_metrics.csv；逐域结果见 domain_metrics.csv；等权域平均见 macro_metrics.csv；对比差值见 comparison_deltas.csv。原始预测、GT、COCO JSON和域指标保留在各实验/国家子目录。']
    (root/'测试结果报告.md').write_text('\n'.join(report)+'\n')
    (root/'verification.json').write_text(json.dumps(dict(complete_evaluations=35,images_per_model=9433,
            total_model_image_inferences=47165,configs_match=True,prediction_files_complete=True),indent=2)+'\n')
    print('\n'.join(report[-7:]))


if __name__=='__main__':main()
