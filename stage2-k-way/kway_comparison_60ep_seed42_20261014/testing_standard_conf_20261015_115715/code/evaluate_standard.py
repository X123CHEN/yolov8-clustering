"""Seven-domain evaluation with the fixed stage-one test metric definitions."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from tqdm import tqdm
from utils.utils import seed_everything
from utils.comparison_evaluation import (ROOT, DOMAINS, StrictYOLO, sha256, write_json,
    voc_metrics, coco_metrics)


from standard_utils_bbox import DecodeBox


class StandardConfidenceYOLO(StrictYOLO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bbox_util = DecodeBox(self.num_classes, self.input_shape)


def verify_standard_confidence():
    decoder = DecodeBox(4, [640, 640])
    prediction = torch.tensor([[[.5, .5, .2, .2, .8, .1, .2, .3],
                                [.1, .1, .1, .1, .01, .01, .01, .01]]])
    result = decoder.non_max_suppression(prediction, 4, [640, 640], [640, 640], False,
                                        conf_thres=.5, nms_thres=.5)[0]
    assert result is not None and len(result) == 1
    assert abs(float(result[0, 4]) - .8) < 1e-6
    print('STANDARD_CONFIDENCE_CHECK_OK score=0.8 threshold=0.5', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only: first N images per domain')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one GPU')
    verify_standard_confidence()
    output = args.output.resolve()
    if (output/'evaluation_config.json').exists():
        raise FileExistsError(f'Choose a new evaluation directory: {output}')
    output.mkdir(parents=True,exist_ok=True)
    seed_everything(args.seed)
    yolo = StandardConfidenceYOLO(model_path=str(args.weights.resolve()),classes_path=str(ROOT/'model_data/rdd_classes.txt'),
                      phi='s',input_shape=[640,640],confidence=.001,nms_iou=.5,letterbox_image=False,cuda=True)
    config = dict(experiment=args.experiment,seed=args.seed,weights=str(args.weights.resolve()),weights_sha256=sha256(args.weights),
                  gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),confidence=.001,nms_iou=.5,score_threshold=.5,
                  confidence_mode='max_class_sigmoid',decoder_sha256=sha256(Path(__file__).with_name('standard_utils_bbox.py')),decoder_path=str(Path(__file__).with_name('standard_utils_bbox.py')),
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


if __name__ == '__main__':
    main()
