"""Extract fixed backbone feat1 statistics from the three original source train lists."""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

import config as C
from common import sha256, save_json, write_csv


def build_manifest():
    rows, seen, manifest_hashes = [], set(), {}
    for country, expected in zip(C.COUNTRIES, C.EXPECTED_COUNTS):
        source = C.MANIFEST_ROOT / f'{country}_train.txt'
        manifest_hashes[str(source)] = sha256(source)
        lines = [(i, s.strip()) for i, s in enumerate(source.read_text().splitlines(), 1) if s.strip()]
        if len(lines) != expected:
            raise ValueError(f'{source}: expected {expected} images, got {len(lines)}')
        country_paths = set()
        for line_no, line in lines:
            if '&' in line:
                raise ValueError('Grouped style-training rows are forbidden')
            parts = line.split()
            path = Path(parts[0]).resolve(strict=True)
            if path in seen:
                raise ValueError(f'Duplicate image path: {path}')
            seen.add(path)
            country_paths.add(path)
            if not path.name.startswith(country + '_'):
                raise ValueError(f'Unexpected source country: {path}')
            boxes = []
            for token in parts[1:]:
                values = [int(v) for v in token.split(',')]
                if len(values) != 5 or values[4] not in range(4):
                    raise ValueError(f'Invalid annotation: {source}:{line_no}')
                boxes.append(values)
            rows.append(dict(row_id=len(rows), image_path=str(path), image_id=path.stem,
                             country=country, split='train', source_txt=str(source),
                             source_line=line_no, boxes_json=json.dumps(boxes),
                             degenerate_box_count=sum(b[0] >= b[2] or b[1] >= b[3] for b in boxes)))
        # Source val paths are used only for overlap auditing, never for fitting/scoring.
        val = C.MANIFEST_ROOT / f'{country}_val.txt'
        manifest_hashes[str(val)] = sha256(val)
        val_paths = {Path(s.split()[0]).resolve() for s in val.read_text().splitlines() if s.strip()}
        if country_paths & val_paths:
            raise ValueError(f'Train/validation overlap in {country}')
    return rows, manifest_hashes


class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]['image_path']) as im:
            im = im.convert('RGB').resize((640, 640), Image.Resampling.BICUBIC)
            x = np.asarray(im, dtype=np.float32) / np.float32(255.0)
        return torch.from_numpy(x.transpose(2, 0, 1).copy())


def load_backbone(device):
    source = C.PROJECT / 'nets/backbone.py'
    spec = importlib.util.spec_from_file_location('latent_analysis_backbone', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Process-local correction: the repository Conv has a redundant hard-coded x.to('cuda').
    # The convolution, BN and activation computation stays identical; source is untouched.
    module.Conv.forward = lambda self, x: self.act(self.bn(self.conv(x)))
    model = module.Backbone(phi='s', pretrained=False)
    state = torch.load(C.WEIGHTS, map_location='cpu', weights_only=True)
    for key in ('state_dict', 'model_state_dict'):
        if key in state:
            state = state[key]
    clean = {k.removeprefix('module.'): v for k, v in state.items()}
    backbone_state = {k[len('backbone.'):]: v for k, v in clean.items() if k.startswith('backbone.')}
    model.load_state_dict(backbone_state, strict=True)
    model.requires_grad_(False).eval().to(device)
    print(f'Strictly loaded {len(backbone_state)} backbone tensors', flush=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    C.ensure_dirs()
    if (C.FEATURES / 'extraction_metadata.json').exists():
        raise RuntimeError('Completed features exist; use a new output directory for a new extraction')
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, hashes = build_manifest()
    write_csv(C.FEATURES / 'metadata.csv', rows)
    device = torch.device(args.device)
    model = load_backbone(device)
    loader = DataLoader(Images(rows), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    partial = C.FEATURES / 'style_features.partial.npy'
    features = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(len(rows), 256))
    start, offset = time.monotonic(), 0
    with torch.inference_mode():
        for step, images in enumerate(loader):
            images = images.to(device, non_blocking=True)
            # P3 := backbone dark3 output (feat1), not the neck P3 variable.
            feat = model.dark3(model.dark2(model.stem(images)))
            if tuple(feat.shape[1:]) != (128, 80, 80):
                raise ValueError(f'Unexpected feature shape {feat.shape}')
            vec = torch.cat((feat.mean((2, 3)), feat.std((2, 3), correction=1)), dim=1)
            arr = vec.cpu().numpy()
            if not np.isfinite(arr).all():
                raise ValueError('Non-finite features')
            features[offset:offset + len(arr)] = arr
            offset += len(arr)
            if step % 25 == 0 or offset == len(rows):
                print(f'features {offset}/{len(rows)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    features.flush()
    del features
    os.replace(partial, C.FEATURES / 'style_features.npy')
    metadata = dict(sample_count=offset, country_counts=dict(zip(C.COUNTRIES, C.EXPECTED_COUNTS)),
                    weights=str(C.WEIGHTS), weights_sha256=sha256(C.WEIGHTS),
                    backbone_sha256=sha256(C.PROJECT / 'nets/backbone.py'),
                    extractor_sha256=sha256(Path(__file__)), manifest_hashes=hashes,
                    phi='s', feature_layer='backbone.dark3 / feat1 / P3', feature_shape=[128,80,80],
                    feature_dimension=256, preprocessing='RGB; direct BICUBIC resize 640x640; float32 /255; no augmentation',
                    std_correction=1, eval_mode=True, frozen=True, precision='float32', seed=0,
                    device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    torch_version=torch.__version__, numpy_version=np.__version__, pillow_version=PIL.__version__,
                    features_sha256=sha256(C.FEATURES / 'style_features.npy'),
                    metadata_sha256=sha256(C.FEATURES / 'metadata.csv'), elapsed_seconds=time.monotonic()-start)
    save_json(C.FEATURES / 'extraction_metadata.json', metadata)
    print('Feature extraction complete.', flush=True)


if __name__ == '__main__':
    main()
