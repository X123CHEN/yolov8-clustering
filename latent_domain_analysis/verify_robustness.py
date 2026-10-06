"""Audit saved indices, all-image predictions, paired ARIs, summaries and source integrity."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'4')
import itertools
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits
import config as C
from common import read_csv, save_json, sha256


OUT=C.RESULTS/'robustness'


def close(a,b):
    assert np.isclose(float(a),float(b),rtol=1e-9,atol=1e-11),(a,b)


def main():
    complete=json.loads((OUT/'complete.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['source_sha256'].items():
        assert sha256(path)==digest,path
    for name,digest in complete['output_sha256'].items():
        assert sha256(OUT/name)==digest,name
    raw=np.load(C.FEATURES/'style_features.npy').astype(float)
    x=(raw-np.load(C.RESULTS/'scaler_mean.npy'))/np.load(C.RESULTS/'scaler_scale.npy')
    n=len(x)
    eval_idx=np.load(OUT/'evaluation_subset_indices.npy')
    fit_idx=np.load(OUT/'fitting_subset_indices.npy')
    fixed=np.load(OUT/'fixed_refit_evaluation_indices.npy')
    assert eval_idx.shape==(10,5000) and fit_idx.shape==(10,10194)
    assert np.array_equal(fixed,np.load(C.RESULTS/'silhouette_sample_indices.npy'))
    for arr,seeds in [(eval_idx,protocol['evaluation_seeds']),(fit_idx,protocol['fitting_subsample_seeds'])]:
        for idx,seed in zip(arr,seeds):
            assert len(set(idx.tolist()))==len(idx) and idx.min()>=0 and idx.max()<n
            expected=np.sort(np.random.default_rng(seed).choice(n,len(idx),replace=False))
            assert np.array_equal(idx,expected)
    eval_runs=read_csv(OUT/'silhouette_subset_runs.csv')
    fit_runs=read_csv(OUT/'subsampling_runs.csv')
    pairs=read_csv(OUT/'subsampling_pairwise_ari.csv')
    occupancy=read_csv(OUT/'subsampling_occupancy.csv')
    expected={(i,k) for i in range(10) for k in C.K_VALUES}
    for rows in [eval_runs,fit_runs]:
        assert len(rows)==70 and {(int(r['repeat']),int(r['K'])) for r in rows}==expected
        for i in range(10):
            group=[r for r in rows if int(r['repeat'])==i]
            for r in group:
                assert int(r['silhouette_rank'])==1+sum(float(s['silhouette'])>float(r['silhouette']) for s in group)
    assert len(pairs)==315
    pair_lookup={(int(r['K']),int(r['repeat_a']),int(r['repeat_b'])):float(r['ARI']) for r in pairs}
    for k in C.K_VALUES:
        data=np.load(OUT/f'subsampling_k{k}.npz')
        labs,centers,mappings=data['all_labels'],data['centers'],data['raw_to_reference']
        ref=np.load(C.RESULTS/'candidates'/f'k{k}.npz')['labels']
        assert labs.shape==(10,n) and centers.shape==(10,k,256)
        for i in range(10):
            ds=(x*x).sum(1)[:,None]+(centers[i]*centers[i]).sum(1)[None,:]-2*x@centers[i].T
            assert np.array_equal(ds.argmin(1),labs[i]),(k,i,'nearest centre')
            row=next(r for r in fit_runs if int(r['K'])==k and int(r['repeat'])==i)
            close(row['fit_inertia'],ds[fit_idx[i],labs[i,fit_idx[i]]].sum())
            close(row['reference_ari'],adjusted_rand_score(ref,labs[i]))
            assert sorted(mappings[i].tolist())==list(range(k))
            aligned=mappings[i][labs[i]]
            close(row['aligned_label_agreement'],np.mean(aligned==ref))
            occ=[r for r in occupancy if int(r['K'])==k and int(r['repeat'])==i]
            assert len(occ)==k
            for r in occ:
                cluster=int(r['reference_cluster'])
                assert int(r['all_count'])==int(np.sum(aligned==cluster))
                assert int(r['fit_count'])==int(np.sum(aligned[fit_idx[i]]==cluster))
                close(r['all_fraction'],np.mean(aligned==cluster))
        for a,b in itertools.combinations(range(10),2):
            close(pair_lookup[k,a,b],adjusted_rand_score(labs[a],labs[b]))
    for runs,filename in [(eval_runs,'silhouette_subset_summary.csv'),(fit_runs,'subsampling_summary.csv')]:
        for summary in read_csv(OUT/filename):
            k=int(summary['K'])
            group=[r for r in runs if int(r['K'])==k]
            for name,value in summary.items():
                for suffix,fn in [('_mean',np.mean),('_std',lambda a:np.std(a,ddof=1)),('_min',np.min),('_max',np.max)]:
                    if name.endswith(suffix):
                        field=name[:-len(suffix)]
                        if field in group[0]:
                            close(value,fn([float(r[field]) for r in group]))
                        elif field=='pairwise_ari':
                            close(value,fn([float(r['ARI']) for r in pairs if int(r['K'])==k]))
    for rows,name in [(eval_runs,'silhouette_k2_k3_differences.csv'),(fit_runs,'subsampling_k2_k3_differences.csv')]:
        for r in read_csv(OUT/name):
            i=int(r['repeat'])
            values={int(t['K']):float(t['silhouette']) for t in rows if int(t['repeat'])==i}
            close(r['delta_k2_minus_k3'],values[2]-values[3])
    # Direct Euclidean computation independently checks the precomputed-distance path.
    for k in [2,3]:
        ref=np.load(C.RESULTS/'candidates'/f'k{k}.npz')['labels']
        row=next(r for r in eval_runs if int(r['K'])==k and int(r['repeat'])==7)
        close(row['silhouette'],silhouette_score(x[eval_idx[7]],ref[eval_idx[7]],metric='euclidean'))
    if C.FINAL_K is None:
        assert not (C.RESULTS/'final_cluster_labels.csv').exists()
    else:
        final=read_csv(C.RESULTS/'final_cluster_labels.csv')
        reviewed=np.load(C.RESULTS/'candidates'/f'k{C.FINAL_K}.npz')['labels']
        assert np.array_equal([int(r['latent_domain']) for r in final],reviewed)
    save_json(OUT/'verification.json',dict(status='passed',evaluation_index_sets=10,fitting_index_sets=10,
              evaluation_scores=70,refits=70,all_image_nearest_center_match=True,
              independently_recomputed_pairwise_ari=315,summary_and_rank_checks=True,
              direct_euclidean_silhouette_spot_checks=2,original_artifacts_unchanged=True,final_k=C.FINAL_K))
    print('PASS: 20 index sets, 70 all-image refits, 315 ARIs, summary/rank checks, 2 independent silhouette checks; original inputs unchanged.')


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
