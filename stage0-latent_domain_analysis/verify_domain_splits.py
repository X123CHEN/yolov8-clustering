"""Independently check TXT coverage, annotations, pairing and validation assignments."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import config as C
from common import read_csv, sha256, save_json

OUT=C.ROOT/'domain_splits'


def lines(path):
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]


def original_mapping(split):
    mapping={}
    for country in C.COUNTRIES:
        for line in lines(C.MANIFEST_ROOT/f'{country}_{split}.txt'):
            path=str(Path(line.split()[0]).resolve())
            assert path not in mapping and Path(path).is_file(),path
            mapping[path]=line
    return mapping


def check():
    summary=json.loads((OUT/'split_summary.json').read_text())
    assert C.FINAL_K==summary['K']==2
    for source,digest in summary['original_imagesets_sha256'].items():
        assert sha256(source)==digest,source
    for name,digest in summary['output_sha256'].items():
        assert sha256(OUT/name)==digest,name
    train=original_mapping('train')
    val=original_mapping('val')
    assert len(train)==12742 and len(val)==1592 and not train.keys()&val.keys()
    groups={}
    for split,original,csvname in [('train',train,'final_cluster_labels.csv'),('val',val,'final_val_cluster_labels.csv')]:
        groups[split]=[lines(OUT/f'Cluster_{k}_{split}.txt') for k in range(2)]
        flat=[s for group in groups[split] for s in group]
        assert len(flat)==len(original) and Counter(flat)==Counter(original.values()),split
        assignments={str(Path(r['image_path']).resolve()):int(r['latent_domain']) for r in read_csv(C.RESULTS/csvname)}
        assert len(assignments)==len(original)
        for k,group in enumerate(groups[split]):
            for line in group:
                path=str(Path(line.split()[0]).resolve())
                assert line==original[path],(path,'annotation altered')
                assert assignments[path]==k,(path,'wrong cluster')
        feature_dir=C.FEATURES if split=='train' else C.FEATURES/'validation'
        meta=read_csv(feature_dir/'metadata.csv')
        x=(np.load(feature_dir/'style_features.npy').astype(float)-np.load(C.RESULTS/'scaler_mean.npy'))/np.load(C.RESULTS/'scaler_scale.npy')
        centers=np.load(C.RESULTS/'cluster_centers.npy')
        nearest=((x[:,None,:]-centers[None,:,:])**2).sum(2).argmin(1)
        assert all(assignments[str(Path(r['image_path']).resolve())]==int(k) for r,k in zip(meta,nearest)),split
    reviewed=np.load(C.RESULTS/'candidates/k2.npz')
    assert np.allclose(np.load(C.RESULTS/'cluster_centers.npy'),reviewed['centers'],rtol=1e-12,atol=1e-12)
    final=read_csv(C.RESULTS/'final_cluster_labels.csv')
    assert np.array_equal(np.array([int(r['latent_domain']) for r in final]),reviewed['labels'])
    combined=lines(OUT/'combined_train.txt')
    assert len(combined)==max(map(len,groups['train']))
    for i,row in enumerate(combined):
        assert row.split('&')==[group[i%len(group)] for group in groups['train']],i
    counts=Counter(str(Path(p.split()[0]).resolve()) for row in combined for p in row.split('&'))
    assert counts.keys()==train.keys()
    repetition=[]
    for k,group in enumerate(groups['train']):
        hist=Counter(counts[str(Path(s.split()[0]).resolve())] for s in group)
        recorded=summary['training_combination']['repetition'][k]
        assert hist==Counter({int(a):int(b) for a,b in recorded['occurrence_histogram'].items()})
        extra=sum(counts[str(Path(s.split()[0]).resolve())]-1 for s in group)
        assert extra==recorded['extra_appearances']
        repetition.append(dict(cluster=k,unique_images=len(group),extra_appearances=extra))
    ungrouped_train=lines(OUT/'all_train.txt')
    ungrouped_val=lines(OUT/'combined_val.txt')
    assert Counter(ungrouped_train)==Counter(train.values())
    assert ungrouped_val==groups['val'][0]+groups['val'][1]
    assert Counter(ungrouped_val)==Counter(val.values()) and all('&' not in s for s in ungrouped_val)
    report=dict(status='passed',K=2,train_images=len(train),validation_images=len(val),
                train_cluster_counts=list(map(len,groups['train'])),validation_cluster_counts=list(map(len,groups['val'])),
                train_val_overlap=0,annotations_preserved=True,single_cluster_duplicates=0,
                combined_train_rows=len(combined),combined_train_missing_images=0,repetition=repetition,
                combined_val_rows=len(ungrouped_val),combined_val_duplicate_images=0,
                validation_training_center_match_rate=1.0,final_train_matches_reviewed_candidate=True,
                original_imagesets_unchanged=True,original_imagesets_files_checked=len(summary['original_imagesets_sha256']))
    save_json(OUT/'verification.json',report)
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':
    check()
