"""Separate evaluation-subset sensitivity from 80% fitting-subsample stability.

Uses frozen source-train features, scaler, and reviewed candidate definitions.
No feature extraction, target data, balancing, final K selection, or training.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import itertools
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, pairwise_distances,
                             silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from threadpoolctl import threadpool_limits

import config as C
from common import read_csv, write_csv, save_json, sha256


EVALUATION_SEEDS = tuple(range(100, 110))
SUBSAMPLING_SEEDS = tuple(range(200, 210))
FIT_SEED = 0
SAMPLE_FRACTION = 0.8
OUT = C.RESULTS / 'robustness'
FIG = C.FIGURES / 'robustness'


def load_inputs():
    extraction = json.loads((C.FEATURES / 'extraction_metadata.json').read_text())
    if sha256(C.FEATURES / 'style_features.npy') != extraction['features_sha256']:
        raise RuntimeError('Feature hash mismatch')
    if sha256(C.FEATURES / 'metadata.csv') != extraction['metadata_sha256']:
        raise RuntimeError('Image-row mapping hash mismatch')
    raw = np.load(C.FEATURES / 'style_features.npy', allow_pickle=False).astype(np.float64)
    mean = np.load(C.RESULTS / 'scaler_mean.npy', allow_pickle=False)
    scale = np.load(C.RESULTS / 'scaler_scale.npy', allow_pickle=False)
    if raw.shape != (12742, 256) or not np.isfinite(raw).all() or not (scale > 0).all():
        raise RuntimeError('Invalid feature/scaler arrays')
    x = (raw - mean) / scale  # Never refit the scaler in either supplementary experiment.
    candidates = {}
    for k in C.K_VALUES:
        with np.load(C.RESULTS / 'candidates' / f'k{k}.npz', allow_pickle=False) as data:
            labels = data['labels'].copy()
        if labels.shape != (len(x),) or set(np.unique(labels)) != set(range(k)):
            raise RuntimeError(f'Invalid original candidate K={k}')
        candidates[k] = labels
    return x, candidates


def distribution(values, prefix):
    values = np.asarray(values, dtype=float)
    return {prefix + '_mean': float(values.mean()), prefix + '_std': float(values.std(ddof=1)),
            prefix + '_min': float(values.min()), prefix + '_max': float(values.max())}


def distance_matrix(x):
    distances = pairwise_distances(x, metric='euclidean')
    np.fill_diagonal(distances, 0.)
    return distances


def score_silhouette(distances, labels, k):
    if len(np.unique(labels)) != k:
        raise RuntimeError(f'Evaluation subset misses a cluster for K={k}; use a uniformly revised protocol')
    return float(silhouette_score(distances, labels, metric='precomputed'))


def rank_rows(rows):
    """Ranks only within a shared evaluation/fitting replicate, using silhouette."""
    for repeat in sorted({r['repeat'] for r in rows}):
        group = [r for r in rows if r['repeat'] == repeat]
        for row in group:
            row['silhouette_rank'] = 1 + sum(r['silhouette'] > row['silhouette'] for r in group)


def paired_differences(rows):
    lookup = {(r['repeat'], r['K']): r['silhouette'] for r in rows}
    return [dict(repeat=i, silhouette_k2=lookup[i, 2], silhouette_k3=lookup[i, 3],
                 delta_k2_minus_k3=lookup[i, 2] - lookup[i, 3])
            for i in sorted({r['repeat'] for r in rows})]


def align_to_reference(labels, reference, k):
    contingency = np.zeros((k, k), dtype=np.int64)
    np.add.at(contingency, (labels, reference), 1)
    src, dst = linear_sum_assignment(-contingency)
    mapping = np.empty(k, dtype=np.int64)
    mapping[src] = dst
    return mapping[labels], mapping


def plot_results(evaluation, subsampling, eval_deltas, fit_deltas, occupancy):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [
        (evaluation, 'silhouette', 'Fixed candidates; 10 evaluation subsets', 'Silhouette', 'silhouette_subsets_vs_k'),
        (subsampling, 'pairwise_ari', '80% subsampling; all-image pairwise ARI', 'ARI', 'subsampling_stability_vs_k'),
        (subsampling, 'reference_ari', '80% subsampling; ARI vs original full fit', 'ARI', 'subsampling_reference_ari_vs_k'),
        (subsampling, 'silhouette', '80% refits; one fixed evaluation subset', 'Silhouette', 'subsampling_silhouette_vs_k'),
    ]
    for ax, (rows, key, title, ylabel, name) in zip(axes.flat, specs):
        for target in [ax]:
            target.errorbar([r['K'] for r in rows], [r[key+'_mean'] for r in rows],
                            yerr=[r[key+'_std'] for r in rows], marker='o', capsize=4)
            target.set(title=title, xlabel='K', ylabel=ylabel, xticks=C.K_VALUES)
            target.grid(alpha=.25)
        single, sa = plt.subplots(figsize=(7, 4.5))
        sa.errorbar([r['K'] for r in rows], [r[key+'_mean'] for r in rows],
                    yerr=[r[key+'_std'] for r in rows], marker='o', capsize=4)
        sa.set(title=title, xlabel='K', ylabel=ylabel, xticks=C.K_VALUES)
        sa.grid(alpha=.25)
        single.tight_layout()
        single.savefig(FIG / f'{name}.png', dpi=160)
        plt.close(single)
    fig.suptitle('Sensitivity checks (error bars: sample SD, not confidence intervals)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(FIG / 'robustness_overview.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, rows, title in zip(axes, [eval_deltas, fit_deltas],
                              ['Change evaluation subset only', 'Change fitting subset only']):
        ax.plot([r['repeat']+1 for r in rows], [r['delta_k2_minus_k3'] for r in rows], 'o-')
        ax.axhline(0, color='black', ls='--')
        ax.set(title=title, xlabel='Repeat', ylabel='Silhouette(K=2) - Silhouette(K=3)', xticks=range(1, 11))
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / 'k2_k3_paired_differences.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, k in zip(axes.flat, C.K_VALUES):
        rows = [r for r in occupancy if r['K'] == k]
        for cluster in range(k):
            values = [r['all_fraction'] for r in rows if r['reference_cluster'] == cluster]
            ax.plot(range(1, 11), values, 'o-', ms=2, label=str(cluster))
        ax.set(title=f'K={k}', xlabel='Subsample repeat', ylabel='All-image cluster fraction')
        ax.grid(alpha=.2)
        ax.legend(fontsize=6, ncol=4)
    axes.flat[-1].axis('off')
    fig.suptitle('Cluster sizes aligned to original labels for display only')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(FIG / 'subsampling_occupancy.png', dpi=160)
    plt.close(fig)


def main():
    if (OUT / 'complete.json').exists():
        raise RuntimeError('Completed robustness results exist; preserve them and use a new experiment directory')
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(OUT/'execution.log', mode='a')])
    started = time.monotonic()
    x, candidates = load_inputs()
    n = len(x)
    n_sub = round(SAMPLE_FRACTION * n)
    eval_indices = np.stack([np.sort(np.random.default_rng(seed).choice(n, 5000, replace=False)) for seed in EVALUATION_SEEDS])
    fit_indices = np.stack([np.sort(np.random.default_rng(seed).choice(n, n_sub, replace=False)) for seed in SUBSAMPLING_SEEDS])
    fixed_eval = np.load(C.RESULTS / 'silhouette_sample_indices.npy', allow_pickle=False)
    np.save(OUT/'evaluation_subset_indices.npy', eval_indices)
    np.save(OUT/'fitting_subset_indices.npy', fit_indices)
    np.save(OUT/'fixed_refit_evaluation_indices.npy', fixed_eval)
    source_files = [C.FEATURES/'style_features.npy', C.FEATURES/'metadata.csv',
                    C.RESULTS/'scaler_mean.npy', C.RESULTS/'scaler_scale.npy',
                    C.RESULTS/'k_selection_metrics.csv', C.RESULTS/'silhouette_sample_indices.npy']
    source_files += [C.RESULTS/'candidates'/f'k{k}.npz' for k in C.K_VALUES]
    source_hashes = {str(p): sha256(p) for p in source_files}
    protocol = dict(n_images=n, feature_dimension=x.shape[1], k_values=C.K_VALUES,
                    evaluation_seeds=EVALUATION_SEEDS, evaluation_size=5000,
                    fitting_subsample_seeds=SUBSAMPLING_SEEDS, fitting_fraction=SAMPLE_FRACTION,
                    fitting_size=n_sub, fitting_size_rule='round(0.8 * N)', sampling='uniform without replacement; no country balancing',
                    same_indices_for_all_k=True, scaler='frozen original full source-train scaler',
                    fit_seed=FIT_SEED, n_init=C.N_INIT, max_iter=C.MAX_ITER, tol=1e-4, algorithm='lloyd',
                    evaluation_subset_experiment='original fixed candidate labels; no refit',
                    subsampling_experiment='refit on 80%; predict all images; pairwise ARI and original-fit ARI',
                    refit_silhouette='use original fixed 5000-image evaluation subset in every repeat; descriptive, not held-out performance',
                    occupancy_alignment='Hungarian maximal overlap with original full-fit labels, for display only; ARI uses raw labels',
                    standard_deviation='ddof=1; descriptive spread, not a confidence interval; repeats/pairs are dependent',
                    numpy_version=np.__version__, sklearn_version=sklearn.__version__, scipy_version=scipy.__version__,
                    source_sha256=source_hashes, script_sha256=sha256(Path(__file__)), final_k=C.FINAL_K)
    save_json(OUT/'protocol.json', protocol)
    evaluation = []
    for repeat, idx in enumerate(eval_indices):
        distances = distance_matrix(x[idx])
        for k in C.K_VALUES:
            evaluation.append(dict(repeat=repeat, seed=EVALUATION_SEEDS[repeat], K=k,
                                   silhouette=score_silhouette(distances, candidates[k][idx], k)))
        logging.info('Evaluation subset %d/10 complete (%.1fs)', repeat+1, time.monotonic()-started)
    del distances
    rank_rows(evaluation)
    eval_deltas = paired_differences(evaluation)
    evaluation_summary = []
    for k in C.K_VALUES:
        rows = [r for r in evaluation if r['K'] == k]
        evaluation_summary.append(dict(K=k, **distribution([r['silhouette'] for r in rows], 'silhouette'),
                                       mean_rank=float(np.mean([r['silhouette_rank'] for r in rows])),
                                       best_count=sum(r['silhouette_rank']==1 for r in rows)))
    write_csv(OUT/'silhouette_subset_runs.csv', evaluation)
    write_csv(OUT/'silhouette_subset_summary.csv', evaluation_summary)
    write_csv(OUT/'silhouette_k2_k3_differences.csv', eval_deltas)
    fixed_distances = distance_matrix(x[fixed_eval])
    metadata = read_csv(C.FEATURES/'metadata.csv')
    write_csv(OUT/'fitting_subset_country_counts.csv',
              [dict(repeat=i, seed=SUBSAMPLING_SEEDS[i], country=country,
                    count=sum(metadata[j]['country']==country for j in idx))
               for i,idx in enumerate(fit_indices) for country in C.COUNTRIES])
    runs, occupancy, pairs, summaries = [], [], [], []
    for k in C.K_VALUES:
        labels_list, centers_list, mappings = [], [], []
        for repeat, idx in enumerate(fit_indices):
            fit_started = time.monotonic()
            model = KMeans(n_clusters=k, init='k-means++', n_init=C.N_INIT,
                           max_iter=C.MAX_ITER, tol=1e-4, random_state=FIT_SEED, algorithm='lloyd').fit(x[idx])
            labels = model.predict(x)
            aligned, mapping = align_to_reference(labels, candidates[k], k)
            labels_list.append(labels)
            centers_list.append(model.cluster_centers_)
            mappings.append(mapping)
            counts = np.bincount(labels, minlength=k)
            row = dict(repeat=repeat, subsampling_seed=SUBSAMPLING_SEEDS[repeat], K=k,
                       fit_seed=FIT_SEED, fit_size=n_sub, fit_inertia=float(model.inertia_),
                       fit_iterations=int(model.n_iter_), reference_ari=float(adjusted_rand_score(candidates[k],labels)),
                       aligned_label_agreement=float(np.mean(aligned==candidates[k])),
                       silhouette=score_silhouette(fixed_distances,labels[fixed_eval],k),
                       ch=float(calinski_harabasz_score(x,labels)), db=float(davies_bouldin_score(x,labels)),
                       min_cluster_fraction=float(counts.min()/n), occupancy_raw=json.dumps(counts.tolist()),
                       elapsed_seconds=time.monotonic()-fit_started)
            runs.append(row)
            for cluster in range(k):
                occupancy.append(dict(K=k, repeat=repeat, reference_cluster=cluster,
                                      all_count=int(np.sum(aligned==cluster)), all_fraction=float(np.mean(aligned==cluster)),
                                      fit_count=int(np.sum(aligned[idx]==cluster)), fit_fraction=float(np.mean(aligned[idx]==cluster))))
            logging.info('K=%d subsample=%d/10 ARI(ref)=%.5f silhouette=%.5f (%.1fs)',
                         k, repeat+1, row['reference_ari'], row['silhouette'],time.monotonic()-started)
        ari_values=[]
        for a,b in itertools.combinations(range(10),2):
            ari=float(adjusted_rand_score(labels_list[a],labels_list[b]))
            pairs.append(dict(K=k, repeat_a=a, repeat_b=b, ARI=ari))
            ari_values.append(ari)
        np.savez_compressed(OUT/f'subsampling_k{k}.npz', all_labels=np.stack(labels_list),
                            centers=np.stack(centers_list), raw_to_reference=np.stack(mappings),
                            fitting_seeds=np.asarray(SUBSAMPLING_SEEDS))
        selected = [r for r in runs if r['K']==k]
        summary=dict(K=k, **distribution(ari_values,'pairwise_ari'))
        for field in ('reference_ari','aligned_label_agreement','silhouette','ch','db','min_cluster_fraction'):
            summary.update(distribution([r[field] for r in selected],field))
        summaries.append(summary)
        write_csv(OUT/'subsampling_runs.csv',runs)
        write_csv(OUT/'subsampling_pairwise_ari.csv',pairs)
        write_csv(OUT/'subsampling_occupancy.csv',occupancy)
        logging.info('K=%d done: mean pairwise ARI %.5f',k,summary['pairwise_ari_mean'])
    rank_rows(runs)
    fit_deltas = paired_differences(runs)
    for summary in summaries:
        group=[r for r in runs if r['K']==summary['K']]
        summary['silhouette_mean_rank']=float(np.mean([r['silhouette_rank'] for r in group]))
        summary['silhouette_best_count']=sum(r['silhouette_rank']==1 for r in group)
    write_csv(OUT/'subsampling_runs.csv',runs)
    write_csv(OUT/'subsampling_summary.csv',summaries)
    write_csv(OUT/'subsampling_k2_k3_differences.csv',fit_deltas)
    plot_results(evaluation_summary,summaries,eval_deltas,fit_deltas,occupancy)
    for path, digest in source_hashes.items():
        if sha256(path)!=digest:
            raise RuntimeError(f'Original artifact changed during robustness check: {path}')
    save_json(OUT/'complete.json',dict(status='complete; awaiting final K confirmation',
              elapsed_seconds=time.monotonic()-started,evaluation_subsets=10,evaluation_scores=len(evaluation),
              subsampling_fits=len(runs),pairwise_ari_count=len(pairs),final_k=C.FINAL_K,
              original_inputs_unchanged=True,
              output_sha256={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.csv','.npz','.npy')}))
    logging.info('Both checks completed; FINAL_K remains %s.',C.FINAL_K)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
