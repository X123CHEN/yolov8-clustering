"""Explicit source-validation COCO AP50; never silently fall back to VOC AP50."""
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from utils.callbacks import EvalCallback
from utils.utils_bbox import DecodeBox
from utils.utils_map import get_coco_map


class ValidationDecodeBox(DecodeBox):
    def __init__(self, num_classes, input_shape, confidence_mode):
        super().__init__(num_classes, input_shape)
        if confidence_mode not in ('legacy_uncertainty', 'max_class_sigmoid'):
            raise ValueError('Unknown validation confidence mode')
        self.confidence_mode = confidence_mode

    def non_max_suppression(self, prediction, num_classes, *args, **kwargs):
        if self.confidence_mode == 'legacy_uncertainty':
            # Reproduce historical relu(sigmoid(logits)) evidence exactly.
            # Encode the historical score at the original winning class so that
            # the common NMS retains identical labels, ordering and filtering.
            prediction = prediction.clone()
            probabilities = prediction[..., 4:4 + num_classes]
            labels = probabilities.argmax(-1, keepdim=True)
            score = 1 - num_classes / (torch.relu(probabilities) + 1).sum(-1, keepdim=True)
            probabilities.zero_().scatter_(-1, labels, score)
        return super().non_max_suppression(prediction, num_classes, *args, **kwargs)


class CocoValidationCallback(EvalCallback):
    def __init__(self, *args, confidence_mode='legacy_uncertainty', **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_mode = confidence_mode
        self.bbox_util = ValidationDecodeBox(self.num_classes, self.input_shape, confidence_mode)

    def on_epoch_end(self, epoch, model_eval, *, force=False, purpose='scheduled', weights_sha256=None):
        if (epoch % self.period and not force) or not self.eval_flag:
            return None  # Missing measurements must never masquerade as this epoch's AP.
        self.net = model_eval
        folder = Path(self.map_out_path)
        for sub in ('ground-truth', 'detection-results'):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        ids = [Path(line.split()[0]).stem for line in self.val_lines]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate validation image IDs would overwrite predictions')
        for line, image_id in tqdm(zip(self.val_lines, ids), total=len(ids), desc='source val COCO AP50'):
            fields = line.split()
            with Image.open(fields[0]) as image:
                self.get_map_txt(image_id, image, self.class_names, str(folder))
            with (folder / 'ground-truth' / f'{image_id}.txt').open('w') as f:
                for box in fields[1:]:
                    x1, y1, x2, y2, category = map(int, box.split(','))
                    f.write(f'{self.class_names[category]} {x1} {y1} {x2} {y2}\n')
        # Deliberately no except->VOC fallback: failing COCO evaluation must fail
        # visibly, rather than changing the meaning of the same CSV column.
        coco = get_coco_map(self.class_names, str(folder))
        value = float(coco[1])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid source validation COCO AP50: {value}')
        self.last_map_50 = value
        self.maps.append(value); self.epoches.append(epoch)
        with (Path(self.log_dir) / 'epoch_map.txt').open('a') as f:
            f.write(str(value) + '\n')
        with (Path(self.log_dir) / 'validation_coco_metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch=epoch, metric='source_val_coco_ap50', value=value,
                                   confidence_mode=self.confidence_mode, confidence_threshold=self.confidence,
                                   images=len(ids), coco_ap50_95=float(coco[0]),
                                   purpose=purpose, weights_sha256=weights_sha256)) + '\n')
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ordered = sorted(zip(self.epoches, self.maps))
        ax.plot([e for e, _ in ordered], [v for _, v in ordered], color='red')
        ax.set(xlabel='Epoch', ylabel='Source validation COCO AP50 (0-1)',
               title=f'Confidence: {self.confidence_mode}')
        ax.grid(True); fig.savefig(Path(self.log_dir) / 'epoch_map.png'); plt.close(fig)
        print('SOURCE_VAL_COCO_AP50', value, 'confidence_mode', self.confidence_mode, flush=True)
        shutil.rmtree(folder)
        return value
