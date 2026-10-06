"""Export K=2 original annotation rows without modifying the RDD2022 dataset.

Training combined file follows 251030_train: cycle shorter domain, '&' separator.
Validation combined file follows 251030_val: one unique image per row.
Validation labels use the training scaler/centres; no fitting on validation.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import config as C
from common import read_csv, write_csv, save_json, sha256

OUT=C.ROOT/'domain_splits'
VAL=C.FEATURES/'validation'


def original_rows(metadata):
    files={}
    rows=[]
    for r in metadata:
        file=Path(r['source_txt'])
        if file not in files:
            files[file]=file.read_text().splitlines()
        line=files[file][int(r['source_line'])-1].strip()
        parts=line.split()
        if '&' in line or Path(parts[0]).resolve()!=Path(r['image_path']).resolve():
            raise RuntimeError(f'Image/index mismatch at {file}:{r["source_line"]}')
        if [list(map(int,b.split(','))) for b in parts[1:]]!=json.loads(r['boxes_json']):
            raise RuntimeError(f'Annotation mismatch at {file}:{r["source_line"]}')
        rows.append(line)
    return rows


def write_lines(path,rows):
    text='\n'.join(rows)+'\n'
    if path.exists():
        if path.read_text()!=text:
            raise RuntimeError(f'Existing partial export differs: {path}')
        return
    with path.open('x',encoding='utf-8',newline='\n') as f:
        f.write(text)


def check_cached_feature_hashes(directory):
    meta=json.loads((directory/'extraction_metadata.json').read_text())
    for filename,key in [('style_features.npy','features_sha256'),('metadata.csv','metadata_sha256')]:
        if sha256(directory/filename)!=meta[key]:
            raise RuntimeError(f'Cache integrity failure {directory/filename}')
    for source,digest in meta['manifest_hashes'].items():
        if sha256(source)!=digest:
            raise RuntimeError(f'Original source manifest changed: {source}')
    return meta


def audit_reference_format():
    report={}
    for split in ['train','val']:
        country_rows=[(C.MANIFEST_ROOT/f'{c}_{split}.txt').read_text().splitlines()
                      for c in ('Japan','India','United_States')]
        ref=C.MANIFEST_ROOT/f'251030_{split}.txt'
        rows=[s.strip() for s in ref.read_text().splitlines() if s.strip()]
        if split=='train':
            expected=['&'.join(parts[i%len(parts)].strip() for parts in country_rows)
                      for i in range(max(map(len,country_rows)))]
            if rows!=expected:
                raise RuntimeError('251030_train no longer follows deterministic cyclic pairing')
            report[split]=dict(reference=str(ref),rows=len(rows),format='cyclic pairing to longest group, separator=&',
                               source_order=['Japan','India','United_States'])
        else:
            expected=[s.strip() for group in country_rows for s in group if s.strip()]
            if any('&' in s for s in rows) or Counter(rows)!=Counter(expected):
                raise RuntimeError('251030_val no longer contains a unique ungrouped source-validation union')
            report[split]=dict(reference=str(ref),rows=len(rows),format='ungrouped unique union, one image per line')
    return report


def export():
    if C.FINAL_K!=2:
        raise RuntimeError('This export requires the user-confirmed FINAL_K=2')
    if (OUT/'split_summary.json').exists():
        raise RuntimeError('Completed export exists; preserve the existing export')
    expected_names={f'Cluster_{k}_{split}.txt' for k in range(2) for split in ['train','val']}
    expected_names.update(['combined_train.txt','combined_val.txt','all_train.txt'])
    if OUT.exists() and any(p.name not in expected_names for p in OUT.iterdir()):
        raise RuntimeError('Unexpected files in partial export directory')
    source_hashes={str(p):sha256(p) for p in sorted(C.MANIFEST_ROOT.rglob('*.txt'))}
    reference=audit_reference_format()
    train_meta=check_cached_feature_hashes(C.FEATURES)
    val_meta=check_cached_feature_hashes(VAL)
    if train_meta['weights_sha256']!=val_meta['weights_sha256']:
        raise RuntimeError('Train/validation extractors use different weights')
    rows=read_csv(C.RESULTS/'final_cluster_labels.csv')
    source_meta=read_csv(C.FEATURES/'metadata.csv')
    if [r['image_path'] for r in rows]!=[r['image_path'] for r in source_meta]:
        raise RuntimeError('Final training rows differ from feature mapping')
    labels=np.array([int(r['latent_domain']) for r in rows])
    mean=np.load(C.RESULTS/'scaler_mean.npy')
    scale=np.load(C.RESULTS/'scaler_scale.npy')
    centers=np.load(C.RESULTS/'cluster_centers.npy')
    if centers.shape!=(2,256):
        raise RuntimeError('Invalid final centres')
    train_x=(np.load(C.FEATURES/'style_features.npy').astype(float)-mean)/scale
    if not np.array_equal(((train_x[:,None,:]-centers[None,:,:])**2).sum(2).argmin(1),labels):
        raise RuntimeError('Training labels do not match final centres')
    reviewed=np.load(C.RESULTS/'candidates/k2.npz')
    if not np.array_equal(labels,reviewed['labels']) or not np.allclose(centers,reviewed['centers'],rtol=1e-12,atol=1e-12):
        raise RuntimeError('Final clustering does not reproduce the confirmed candidate')
    validation=read_csv(VAL/'metadata.csv')
    val_x=(np.load(VAL/'style_features.npy').astype(float)-mean)/scale
    if val_x.shape!=(1592,256):
        raise RuntimeError('Invalid validation feature shape')
    val_d2=((val_x[:,None,:]-centers[None,:,:])**2).sum(2)
    val_labels=val_d2.argmin(1)
    train_lines=original_rows(source_meta)
    val_lines=original_rows(validation)
    train_paths={r['image_path'] for r in source_meta}
    val_paths={r['image_path'] for r in validation}
    if len(train_paths)!=12742 or len(val_paths)!=1592 or train_paths&val_paths:
        raise RuntimeError('Duplicate or crossing training/validation image paths')
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(C.RESULTS/'final_val_cluster_labels.csv',
              [dict(**r,latent_domain=int(label),distance_sq=float(ds[label]),
                    distance_sq_cluster0=float(ds[0]),distance_sq_cluster1=float(ds[1]))
               for r,label,ds in zip(validation,val_labels,val_d2)])
    write_csv(C.RESULTS/'val_cluster_composition.csv',
              [dict(country=country,cluster=k,count=int(sum(r['country']==country and label==k for r,label in zip(validation,val_labels))))
               for country in C.COUNTRIES for k in range(2)])
    grouped={}
    for split,lines,labs in [('train',train_lines,labels),('val',val_lines,val_labels)]:
        grouped[split]=[[line for line,label in zip(lines,labs) if label==k] for k in range(2)]
        if any(not group for group in grouped[split]):
            raise RuntimeError(f'Empty cluster in {split}')
        for k,group in enumerate(grouped[split]):
            write_lines(OUT/f'Cluster_{k}_{split}.txt',group)
    write_lines(OUT/'all_train.txt',train_lines)
    length=max(map(len,grouped['train']))
    combined_train=['&'.join(group[i%len(group)] for group in grouped['train']) for i in range(length)]
    write_lines(OUT/'combined_train.txt',combined_train)
    write_lines(OUT/'combined_val.txt',[line for group in grouped['val'] for line in group])
    cluster_summary=[]
    for split,labs,meta in [('train',labels,source_meta),('val',val_labels,validation)]:
        for k in range(2):
            cluster_summary.append(dict(split=split,cluster=k,count=int(np.sum(labs==k)),
                                       country_counts={country:int(sum(r['country']==country and label==k for r,label in zip(meta,labs)))
                                                       for country in C.COUNTRIES}))
    repetition=[]
    for k,group in enumerate(grouped['train']):
        counts=Counter(group[i%len(group)].split()[0] for i in range(length))
        repetition.append(dict(cluster=k,unique_images=len(group),appearances=length,
                               extra_appearances=length-len(group),missing_images=0,
                               occurrence_histogram=dict(sorted(Counter(counts.values()).items()))))
    for path,digest in source_hashes.items():
        if sha256(path)!=digest:
            raise RuntimeError(f'Original ImageSets file changed: {path}')
    summary=dict(K=2,training_count=len(train_lines),validation_count=len(val_lines),
                 clusters=cluster_summary,reference_formats=reference,
                 training_combination=dict(rows=length,cluster_order=[0,1],separator='&',
                     rule='deterministic modulo cycling to the longer cluster length; no shuffle; no truncation',
                     repetition=repetition),
                 validation_combination=dict(rows=len(val_lines),rule='unique concatenation: Cluster_0 then Cluster_1',
                     duplicate_images=0,missing_images=0),
                 all_train=dict(rows=len(train_lines),rule='unique source order India, Japan, United_States'),
                 validation_assignment='training scaler and training centres only; no validation fitting',
                 labels_match_reviewed_candidate=True,final_fit_seed=int(reviewed['seed']),
                 original_images_referenced_only=True,source_images_copied_or_moved=False,
                 original_imagesets_sha256=source_hashes,original_imagesets_unchanged=True,
                 validation_feature_metadata_sha256=sha256(VAL/'extraction_metadata.json'),
                 scaler_mean_sha256=sha256(C.RESULTS/'scaler_mean.npy'),
                 scaler_scale_sha256=sha256(C.RESULTS/'scaler_scale.npy'),
                 training_centres_sha256=sha256(C.RESULTS/'cluster_centers.npy'),
                 output_sha256={p.name:sha256(p) for p in sorted(OUT.glob('*.txt'))})
    save_json(OUT/'split_summary.json',summary)
    save_json(C.RESULTS/'final_k_decision.json',dict(final_k=2,confirmed_by='user',
              evidence=['initial K selection','10 evaluation subsets','10 80% subsampling fits'],
              fit_seed=int(reviewed['seed']),validation_fitted=False,export_directory=str(OUT)))
    print(json.dumps(dict(clusters=cluster_summary,combined_train_rows=length,
                         repetition=repetition,output_directory=str(OUT)),indent=2,ensure_ascii=False))


if __name__=='__main__':
    export()
