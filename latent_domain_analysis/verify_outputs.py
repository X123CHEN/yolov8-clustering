"""Independent integrity and nearest-centre checks for the saved K-selection artifacts."""
import json
from collections import Counter
import numpy as np
import config as C
from common import read_csv, sha256, save_json


def main():
    meta=json.loads((C.FEATURES/'extraction_metadata.json').read_text())
    assert sha256(C.WEIGHTS)==meta['weights_sha256']
    for path,digest in meta['manifest_hashes'].items():
        assert sha256(path)==digest, path
    assert sha256(C.FEATURES/'style_features.npy')==meta['features_sha256']
    assert sha256(C.FEATURES/'metadata.csv')==meta['metadata_sha256']
    rows=read_csv(C.FEATURES/'metadata.csv')
    assert Counter(r['country'] for r in rows)==dict(zip(C.COUNTRIES,C.EXPECTED_COUNTS))
    assert len({r['image_path'] for r in rows})==sum(C.EXPECTED_COUNTS)
    assert {r['split'] for r in rows}=={'train'}
    raw=np.load(C.FEATURES/'style_features.npy').astype(np.float64)
    assert raw.shape==(12742,256) and np.isfinite(raw).all()
    mean=np.load(C.RESULTS/'scaler_mean.npy')
    scale=np.load(C.RESULTS/'scaler_scale.npy')
    assert np.isfinite(scale).all() and (scale>0).all()
    x=(raw-mean)/scale
    assert np.allclose(x.mean(0),0,atol=1e-10)
    runs=read_csv(C.RESULTS/'k_selection_runs.csv')
    assert {(int(r['K']),int(r['seed'])) for r in runs}=={(k,s) for k in C.K_VALUES for s in C.SEEDS}
    assert len(read_csv(C.RESULTS/'stability_pairs.csv'))==70
    metrics=read_csv(C.RESULTS/'k_selection_metrics.csv')
    assert len(metrics)==7 and all(r['gmm_converged']=='True' for r in metrics)
    idx=np.load(C.RESULTS/'silhouette_sample_indices.npy')
    assert len(idx)==5000 and len(set(idx))==5000 and idx.min()>=0 and idx.max()<len(x)
    checks=[]
    for row in metrics:
        k=int(row['K'])
        data=np.load(C.RESULTS/'candidates'/f'k{k}.npz')
        labels,centers=data['labels'],data['centers']
        d2=((x[:,None,:]-centers[None,:,:])**2).sum(2)
        assert np.array_equal(d2.argmin(1),labels), f'K={k} nearest-centre mismatch'
        assert np.allclose(d2[np.arange(len(x)),labels].sum(),float(data['inertia']),rtol=1e-10)
        assert np.array_equal(np.bincount(labels,minlength=k),json.loads(row['cluster_occupancy']))
        assert np.isclose(float(data['inertia']),min(float(r['inertia']) for r in runs if int(r['K'])==k))
        assert (C.FIGURES/f'silhouette_profile_k{k}.png').is_file()
        checks.append(dict(K=k,nearest_center_match_rate=1.0,sample_count=len(labels)))
    if C.FINAL_K is None:
        assert not (C.RESULTS/'final_cluster_labels.csv').exists()
    else:
        final=read_csv(C.RESULTS/'final_cluster_labels.csv')
        reviewed=np.load(C.RESULTS/'candidates'/f'k{C.FINAL_K}.npz')['labels']
        assert np.array_equal([int(r['latent_domain']) for r in final],reviewed)
    save_json(C.RESULTS/'verification.json',dict(status='passed',sample_count=len(x),dimension=256,
              kmeans_runs=len(runs),stability_pairs=70,all_gmm_converged=True,candidate_checks=checks,
              final_k=C.FINAL_K,degenerate_annotation_count=sum(int(r['degenerate_box_count']) for r in rows),
              caveat='Path uniqueness checked; no image-content deduplication'))
    print(f'PASS: source manifest and feature integrity, 35 runs, 70 ARI pairs, all candidate nearest-centre assignments; FINAL_K={C.FINAL_K}.')


if __name__=='__main__':
    main()
