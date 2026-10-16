"""Strict checkpoint inference and separately named VOC/COCO test metrics."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from nets.yolo import YoloBody
from utils.utils import cvtColor, preprocess_input, resize_image, seed_everything
from utils.utils_map import voc_ap
from yolo import YOLO

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ['Japan', 'India', 'United_States', 'Norway', 'new_Czech', 'new_China_Drone', 'new_China_MotorBike']
TARGETS = DOMAINS[3:]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


class StrictYOLO(YOLO):
    def generate(self, onnx=False):
        self.net = YoloBody(self.input_shape, self.num_classes, self.phi, pretrained=False)
        state = torch.load(self.model_path, map_location='cpu')
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.net.load_state_dict(state, strict=True)
        self.net = self.net.fuse().eval().cuda()
        print('STRICT_LOAD_OK', self.model_path, 'keys=', len(state), flush=True)

    @torch.no_grad()
    def predict(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        resized = resize_image(image, (640, 640), self.letterbox_image)
        data = np.expand_dims(np.transpose(preprocess_input(np.asarray(resized, dtype=np.float32)), (2, 0, 1)), 0)
        outputs = self.bbox_util.decode_box(self.net(torch.from_numpy(data).cuda()))
        results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                    image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
        if results[0] is None:
            return []
        rows = results[0].detach().cpu().numpy() if torch.is_tensor(results[0]) else np.asarray(results[0])
        predictions = []
        for top, left, bottom, right, score, category in rows:
            if not np.isfinite([top, left, bottom, right, score]).all():
                raise ValueError('Nonfinite prediction')
            if right <= left or bottom <= top:
                continue
            predictions.append(dict(category=int(category), score=float(score),
                                    bbox=[float(left), float(top), float(right), float(bottom)]))
        return predictions


def voc_metrics(ground_truth, predictions, classes, score_threshold=0.5):
    """VOC continuous AP@.5 with inclusive pixel IoU; fixed-threshold P/R/F1.

    Difficult matches are ignored. No predictions above the threshold means
    precision=recall=F1=0, independently of low-score predictions used for AP.
    """
    result = {}
    for category, name in enumerate(classes):
        gt = {image: [obj for obj in objects if obj['category'] == category] for image, objects in ground_truth.items()}
        positives = sum(not obj['difficult'] for objects in gt.values() for obj in objects)
        used = {image: np.zeros(len(objects), dtype=bool) for image, objects in gt.items()}
        detections = [(image, obj) for image, objects in predictions.items() for obj in objects if obj['category'] == category]
        detections.sort(key=lambda item: -item[1]['score'])
        tp, fp, scores = [], [], []
        for image, prediction in detections:
            objects = gt[image]
            best_iou, match = -1., -1
            if objects:
                boxes = np.asarray([obj['bbox'] for obj in objects], dtype=float)
                box = np.asarray(prediction['bbox'])
                wh = np.maximum(0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2]) + 1)
                intersection = wh.prod(1)
                area = ((boxes[:, 2:] - boxes[:, :2] + 1).clip(0)).prod(1)
                other = ((box[2:] - box[:2] + 1).clip(0)).prod()
                iou = intersection / np.maximum(area + other - intersection, 1e-12)
                match = int(iou.argmax()); best_iou = float(iou[match])
            if best_iou >= 0.5 and objects[match]['difficult']:
                continue
            hit = best_iou >= 0.5 and not used[image][match]
            if hit:
                used[image][match] = True
            tp.append(int(hit)); fp.append(int(not hit)); scores.append(prediction['score'])
        cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
        recall = cumulative_tp / max(positives, 1)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
        ap = float(voc_ap(recall.tolist(), precision.tolist())[0]) if positives else None
        kept = np.asarray(scores) >= score_threshold
        true_positive = int(np.asarray(tp)[kept].sum())
        false_positive = int(np.asarray(fp)[kept].sum())
        p = true_positive / max(true_positive + false_positive, 1)
        r = true_positive / max(positives, 1)
        result[name] = dict(ap50=ap, precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.,
                            gt_objects=positives, true_positive=true_positive, false_positive=false_positive,
                            false_negative=positives-true_positive)
    valid = [r for r in result.values() if r['ap50'] is not None]
    return dict(map50=float(np.mean([r['ap50'] for r in valid])) if valid else None,
                per_class=result, precision_macro=float(np.mean([r['precision'] for r in valid])) if valid else None,
                recall_macro=float(np.mean([r['recall'] for r in valid])) if valid else None,
                f1_macro=float(np.mean([r['f1'] for r in valid])) if valid else None)


def coco_metrics(gt, predictions, classes, sizes, destination):
    ids = {name: index + 1 for index, name in enumerate(gt)}
    data = dict(info={'description': 'RDD2022 fixed test split'}, licenses=[],
                images=[dict(id=ids[name], file_name=name+'.jpg', width=sizes[name][0], height=sizes[name][1]) for name in gt],
                categories=[dict(id=i+1, name=name) for i, name in enumerate(classes)], annotations=[])
    detected = []
    for image, objects in gt.items():
        for obj in objects:
            x1,y1,x2,y2 = obj['bbox']
            data['annotations'].append(dict(id=len(data['annotations'])+1,image_id=ids[image],category_id=obj['category']+1,
                    bbox=[x1,y1,x2-x1,y2-y1], area=(x2-x1)*(y2-y1), iscrowd=int(obj['difficult'])))
        for obj in predictions[image]:
            x1,y1,x2,y2 = obj['bbox']
            detected.append(dict(image_id=ids[image],category_id=obj['category']+1,bbox=[x1,y1,x2-x1,y2-y1],score=obj['score']))
    write_json(destination/'coco_ground_truth.json', data)
    write_json(destination/'coco_detections.json', detected)
    coco = COCO(); coco.dataset = data; coco.createIndex()
    if detected:
        dt = coco.loadRes(detected)
    else:
        dt = COCO(); dt.dataset = dict(images=data['images'],categories=data['categories'],annotations=[]); dt.createIndex()
    evaluation = COCOeval(coco, dt, 'bbox')
    evaluation.params.imgIds = list(ids.values())
    evaluation.evaluate(); evaluation.accumulate(); evaluation.summarize()
    precision = evaluation.eval['precision']
    per_class = {}
    for i,name in enumerate(classes):
        all_iou = precision[:,:,i,0,-1]; iou50 = precision[0,:,i,0,-1]
        per_class[name] = dict(ap50_95=float(all_iou[all_iou>=0].mean()) if (all_iou>=0).any() else None,
                               ap50=float(iou50[iou50>=0].mean()) if (iou50>=0).any() else None)
    return dict(map50_95=float(evaluation.stats[0]),map50=float(evaluation.stats[1]),
                map75=float(evaluation.stats[2]),per_class=per_class,max_detections=100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only: first N images per domain')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one GPU')
    output = args.output.resolve()
    if (output/'evaluation_config.json').exists():
        raise FileExistsError(f'Choose a new evaluation directory: {output}')
    output.mkdir(parents=True,exist_ok=True)
    seed_everything(11)
    yolo = StrictYOLO(model_path=str(args.weights.resolve()),classes_path=str(ROOT/'model_data/rdd_classes.txt'),
                      phi='s',input_shape=[640,640],confidence=.001,nms_iou=.5,letterbox_image=False,cuda=True)
    config = dict(experiment=args.experiment,weights=str(args.weights.resolve()),weights_sha256=sha256(args.weights),
                  gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),confidence=.001,nms_iou=.5,score_threshold=.5,
                  input_shape=[640,640],letterbox=False,tta=False,phi='s',classes=yolo.class_names,
                  smoke_limit=args.limit,strict_load=True,domains=DOMAINS,coordinates='full precision xyxy; VOC inclusive IoU, COCO continuous xywh',
                  test_list_sha256={d:sha256(ROOT/'datasets/RDD2022'/d/'ImageSets/Main/test_all.txt') for d in DOMAINS})
    write_json(output/'evaluation_config.json',config)
    summary = []
    for domain in DOMAINS:
        started=time.time();base=ROOT/'datasets/RDD2022'/domain
        image_ids=(base/'ImageSets/Main/test_all.txt').read_text().split()
        if len(set(image_ids))!=len(image_ids):
            raise ValueError(f'Duplicate test IDs: {domain}')
        if args.limit:
            image_ids=image_ids[:args.limit]
        folder=output/domain
        for sub in ['ground-truth','detection-results']:(folder/sub).mkdir(parents=True,exist_ok=True)
        gt,predictions,sizes={},{},{}
        for image_id in tqdm(image_ids,desc=domain,mininterval=10):
            annotation=ET.parse(base/'Annotations'/f'{image_id}.xml').getroot()
            objects=[]
            for obj in annotation.findall('object'):
                name=obj.findtext('name')
                if name not in yolo.class_names:
                    raise ValueError(f'Unknown label: {name}')
                box=[float(obj.find('bndbox').findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
                if box[2]<=box[0] or box[3]<=box[1]:
                    raise ValueError(f'Invalid box in {domain}/{image_id}')
                objects.append(dict(category=yolo.class_names.index(name),bbox=box,difficult=obj.findtext('difficult','0')=='1'))
            gt[image_id]=objects
            with Image.open(base/'JPEGImages'/f'{image_id}.jpg') as image:
                sizes[image_id]=image.size
                predictions[image_id]=yolo.predict(image)
            with (folder/'ground-truth'/f'{image_id}.txt').open('w') as f:
                for obj in objects:
                    f.write(yolo.class_names[obj['category']]+' '+' '.join(map(str,obj['bbox']))+(' difficult' if obj['difficult'] else '')+'\n')
            with (folder/'detection-results'/f'{image_id}.txt').open('w') as f:
                for obj in predictions[image_id]:
                    f.write(yolo.class_names[obj['category']]+' '+str(obj['score'])+' '+' '.join(map(str,obj['bbox']))+'\n')
        voc=voc_metrics(gt,predictions,yolo.class_names)
        coco=coco_metrics(gt,predictions,yolo.class_names,sizes,folder)
        result=dict(experiment=args.experiment,domain=domain,images=len(image_ids),voc=voc,coco=coco,elapsed_s=time.time()-started)
        write_json(folder/'metrics.json',result)
        summary.append(dict(experiment=args.experiment,domain=domain,images=len(image_ids),voc_map50=voc['map50'],
                            coco_map50=coco['map50'],coco_map50_95=coco['map50_95'],
                            precision_macro=voc['precision_macro'],recall_macro=voc['recall_macro'],f1_macro=voc['f1_macro']))
        with (output/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        print('DOMAIN_COMPLETE '+json.dumps(summary[-1]),flush=True)
    write_json(output/'completion.json',dict(experiment=args.experiment,domains_completed=len(summary),
                 images=sum(row['images'] for row in summary),smoke_test=bool(args.limit)))

