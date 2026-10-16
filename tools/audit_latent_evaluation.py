"""Read-only checkpoint audit: replay historical/standard validation and full tests."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from nets.yolo import YoloBody
from utils.callbacks import EvalCallback
from utils.comparison_evaluation import voc_metrics
from utils.comparison_training import prepare_data, file_hash
from utils.utils import get_classes, seed_everything
from utils.utils_map import get_coco_map, get_map


def read_boxes(folder, classes):
    gt, predictions = {}, {}
    for path in sorted((folder / 'ground-truth').glob('*.txt')):
        gt[path.stem] = []
        for line in path.read_text().splitlines():
            fields = line.split()
            gt[path.stem].append(dict(category=classes.index(fields[0]), bbox=list(map(float, fields[1:5])), difficult=False))
        predictions[path.stem] = []
        for line in (folder / 'detection-results' / path.name).read_text().splitlines():
            fields = line.split()
            predictions[path.stem].append(dict(category=classes.index(fields[0]), score=float(fields[1]), bbox=list(map(float, fields[2:6]))))
    return gt, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=['hard_expert','soft_expert','hard_kway','soft_kway'])
    parser.add_argument('--output', type=Path, default=ROOT / 'results/evaluation_audit_20261007')
    args = parser.parse_args()
    name, audit = args.experiment, args.output.resolve()
    output = audit / name
    output.mkdir(parents=True, exist_ok=False)
    suite = ROOT / 'results/latent_suite_20261006_offline'
    run = ROOT / 'results/comparison_20261005_100ep/latent_unique' if name=='hard_expert' else suite/'training'/name
    prior_test = Path(json.loads((suite/'status.json').read_text())['evaluations'][name])
    config = json.loads((run/'run_config.json').read_text())
    test_config = json.loads((prior_test/'evaluation_config.json').read_text())
    best = json.loads((run/'best_checkpoint.json').read_text())
    weights = run/'best_epoch_weights.pth'
    assert file_hash(weights) == test_config['weights_sha256']
    assert file_hash(ROOT/'utils/utils_bbox.py') == test_config['decoder_sha256']
    for domain, digest in test_config['test_list_sha256'].items():
        assert file_hash(ROOT/'datasets/RDD2022'/domain/'ImageSets/Main/test_all.txt') == digest
    _, train_path, val_path, _, records, _, val_lines, _ = prepare_data('latent_unique')
    assert file_hash(train_path) == config['train_sha256'] and file_hash(val_path) == config['val_sha256']
    assert hashlib.sha256('\n'.join(val_lines).encode()).hexdigest() == config['validation_order_sha256']
    ids = [Path(row.split()[0]).stem for row in val_lines]
    assert len(set(ids)) == len(ids) == 1592
    validation_paths = {str(Path(row.split()[0]).resolve()) for row in val_lines}
    overlap = {}
    for domain in test_config['domains']:
        base=ROOT/'datasets/RDD2022'/domain
        test_paths={str((base/'JPEGImages'/f'{key}.jpg').resolve()) for key in (base/'ImageSets/Main/test_all.txt').read_text().split()}
        overlap[domain] = dict(train=len(test_paths & set(records)), val=len(test_paths & validation_paths))
        assert not test_paths & (set(records) | validation_paths)
    history = list(csv.DictReader((run/'metrics/metrics_epoch.csv').open()))
    logged = float(next(row['map_50'] for row in history if int(row['epoch'])==best['epoch']))
    assert min(history,key=lambda row:float(row['val_loss']))['epoch'] == str(best['epoch'])
    seed_everything(11)
    classes, count = get_classes(str(ROOT/'model_data/rdd_classes.txt'))
    model = YoloBody([640,640],count,'s',pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(weights,map_location='cpu'),strict=True)
    spec = importlib.util.spec_from_file_location('audit_legacy_decoder', audit/'legacy_utils_bbox.py')
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    callbacks = {}
    for mode in ['legacy_uncertainty','standard_class']:
        folder=output/mode
        for sub in ['ground-truth','detection-results']:
            (folder/sub).mkdir(parents=True)
        callback=EvalCallback(model,[640,640],classes,count,val_lines,str(output),True,
                              map_out_path=str(folder),eval_flag=False)
        if mode=='legacy_uncertainty':
            callback.bbox_util=legacy.DecodeBox(count,[640,640])
        callbacks[mode]=callback
    for row in tqdm(val_lines,desc=name+' same-val dual-confidence',mininterval=15):
        fields=row.split(); image_id=Path(fields[0]).stem
        gt_lines=[]
        for box in fields[1:]:
            x1,y1,x2,y2,label=map(int,box.split(','))
            gt_lines.append(f'{classes[label]} {x1} {y1} {x2} {y2}')
        with Image.open(fields[0]) as image:
            for mode, callback in callbacks.items():
                folder=output/mode
                callback.get_map_txt(image_id,image,classes,str(folder))
                (folder/'ground-truth'/f'{image_id}.txt').write_text('\n'.join(gt_lines)+'\n')
    result=dict(experiment=name,best_epoch=best['epoch'],logged_map50_at_best=logged,
                weights_sha256=file_hash(weights),decoder_sha256=file_hash(ROOT/'utils/utils_bbox.py'),
                legacy_decoder_sha256=file_hash(audit/'legacy_utils_bbox.py'),
                validation_images=1592,train_test_val_path_overlap=overlap,modes={})
    for mode in callbacks:
        folder=output/mode
        metrics=get_coco_map(classes,str(folder))
        gt,predictions=read_boxes(folder,classes)
        voc=voc_metrics(gt,predictions,classes)['map50']
        result['modes'][mode]=dict(coco50=float(metrics[1]),coco50_95=float(metrics[0]),voc50=voc,
                                  difference_from_training_log=float(metrics[1])-logged)
    (output/'validation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VALIDATION_AUDIT',json.dumps(result),flush=True)
    # Free checkpoint VRAM before the independent full-test process.
    callbacks.clear(); del callback,model; torch.cuda.empty_cache()
    with (output/'test_replay.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'get_map_new.py'),'--weights',str(weights),
                        '--output',str(output/'test_replay'),'--experiment',name],cwd=ROOT,check=True,
                       stdout=log,stderr=subprocess.STDOUT)
    old_rows=list(csv.DictReader((prior_test/'summary.csv').open()))
    new_rows=list(csv.DictReader((output/'test_replay/summary.csv').open()))
    comparisons=[]
    for old,new in zip(old_rows,new_rows):
        assert old['domain']==new['domain'] and old['images']==new['images']
        domain=old['domain']; replay=output/'test_replay'/domain
        # Independently recompute VOC using the original repository implementation.
        independent=get_map(.5,False,path=str(replay))
        difference=abs(independent-float(new['voc_map50']))
        if difference>1e-10:
            raise AssertionError(f'Independent VOC implementation mismatch: {domain} {difference}')
        deltas={m:float(new[m])-float(old[m]) for m in ['voc_map50','coco_map50','coco_map50_95','precision_macro','recall_macro','f1_macro']}
        comparisons.append(dict(domain=domain,deltas=deltas,independent_voc_delta=difference,
                                gt_json_equal=file_hash(prior_test/domain/'coco_ground_truth.json')==file_hash(replay/'coco_ground_truth.json'),
                                detections_json_equal=file_hash(prior_test/domain/'coco_detections.json')==file_hash(replay/'coco_detections.json')))
    result['test_replay']=comparisons
    result['complete']=True
    (output/'audit_complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print('AUDIT_COMPLETE',name,flush=True)


if __name__=='__main__':
    main()
