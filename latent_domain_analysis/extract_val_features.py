"""Extract source-validation feat1 with the frozen training extractor; never fit clusters."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import config as C
from common import sha256, save_json, write_csv
from extract_features import Images, load_backbone

OUT = C.FEATURES / 'validation'
EXPECTED_COUNTS = (322, 790, 480)


def manifest():
    rows,seen,hashes=[],set(),{}
    train_paths=set()
    for country in C.COUNTRIES:
        train=C.MANIFEST_ROOT/f'{country}_train.txt'
        train_paths.update(Path(s.split()[0]).resolve() for s in train.read_text().splitlines() if s.strip())
    for country,expected in zip(C.COUNTRIES,EXPECTED_COUNTS):
        source=C.MANIFEST_ROOT/f'{country}_val.txt'
        hashes[str(source)]=sha256(source)
        lines=[(i,s.strip()) for i,s in enumerate(source.read_text().splitlines(),1) if s.strip()]
        if len(lines)!=expected:
            raise ValueError(f'Unexpected validation count: {source}')
        for line_no,line in lines:
            if '&' in line:
                raise ValueError('Validation input must be an original single-image annotation row')
            parts=line.split()
            path=Path(parts[0]).resolve(strict=True)
            if path in seen or path in train_paths:
                raise ValueError(f'Duplicate or train/val-overlap path: {path}')
            if not path.name.startswith(country+'_'):
                raise ValueError(f'Unexpected source country: {path}')
            seen.add(path)
            boxes=[]
            for token in parts[1:]:
                values=[int(v) for v in token.split(',')]
                if len(values)!=5 or values[4] not in range(4):
                    raise ValueError(f'Invalid annotation: {source}:{line_no}')
                boxes.append(values)
            rows.append(dict(row_id=len(rows),image_path=str(path),image_id=path.stem,country=country,
                             split='val',source_txt=str(source),source_line=line_no,boxes_json=json.dumps(boxes),
                             degenerate_box_count=sum(b[0]>=b[2] or b[1]>=b[3] for b in boxes)))
    return rows,hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if (OUT/'extraction_metadata.json').exists():
        raise RuntimeError('Completed validation features exist; preserve the cache')
    OUT.mkdir(parents=True,exist_ok=True)
    train_meta=json.loads((C.FEATURES/'extraction_metadata.json').read_text())
    if sha256(C.WEIGHTS)!=train_meta['weights_sha256'] or sha256(C.PROJECT/'nets/backbone.py')!=train_meta['backbone_sha256']:
        raise RuntimeError('Extractor weights/backbone changed since training extraction')
    if sha256(C.ROOT/'extract_features.py')!=train_meta['extractor_sha256']:
        raise RuntimeError('Training extraction code changed; verify preprocessing before reuse')
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rows,hashes=manifest()
    write_csv(OUT/'metadata.csv',rows)
    device=torch.device(args.device)
    model=load_backbone(device)
    loader=DataLoader(Images(rows),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                      pin_memory=device.type=='cuda')
    started=time.monotonic()
    chunks=[]
    with torch.inference_mode():
        for i,images in enumerate(loader):
            feat=model.dark3(model.dark2(model.stem(images.to(device,non_blocking=True))))
            if tuple(feat.shape[1:])!=(128,80,80):
                raise ValueError(f'Unexpected feat1 shape {feat.shape}')
            arr=torch.cat([feat.mean((2,3)),feat.std((2,3),correction=1)],1).cpu().numpy()
            if not np.isfinite(arr).all():
                raise ValueError('Non-finite validation feature')
            chunks.append(arr)
            if i%25==0:
                print(f'validation {sum(len(a) for a in chunks)}/{len(rows)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    features=np.concatenate(chunks)
    assert features.shape==(sum(EXPECTED_COUNTS),256)
    partial=OUT/'style_features.partial.npy'
    np.save(partial,features)
    os.replace(partial,OUT/'style_features.npy')
    save_json(OUT/'extraction_metadata.json',dict(sample_count=len(rows),country_counts=dict(zip(C.COUNTRIES,EXPECTED_COUNTS)),
              manifest_hashes=hashes,weights_sha256=train_meta['weights_sha256'],
              feature_layer=train_meta['feature_layer'],feature_dimension=256,
              preprocessing=train_meta['preprocessing'],std_correction=1,
              training_extraction_metadata_sha256=sha256(C.FEATURES/'extraction_metadata.json'),
              validation_extractor_sha256=sha256(Path(__file__)),device=str(device),torch_version=torch.__version__,
              features_sha256=sha256(OUT/'style_features.npy'),metadata_sha256=sha256(OUT/'metadata.csv'),
              fitting_performed=False,elapsed_seconds=time.monotonic()-started))
    print(f'Validation extraction complete: {features.shape}',flush=True)


if __name__=='__main__':
    main()
