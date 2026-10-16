"""Source-only K comparison; deliberately never sets FINAL_K or launches training."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import itertools
import json
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_samples, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score, pairwise_distances)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import config as C
from common import sha256, save_json, write_csv, read_csv


def load_features(save_scaler=False):
    meta = json.loads((C.FEATURES / 'extraction_metadata.json').read_text())
    for filename, key in [('style_features.npy', 'features_sha256'), ('metadata.csv', 'metadata_sha256')]:
        if sha256(C.FEATURES / filename) != meta[key]:
            raise ValueError(f'Feature cache integrity mismatch: {filename}')
    raw = np.load(C.FEATURES / 'style_features.npy', allow_pickle=False)
    if raw.shape != (sum(C.EXPECTED_COUNTS), 256) or not np.isfinite(raw).all():
        raise ValueError(f'Invalid features {raw.shape}')
    scaler = StandardScaler()
    x = scaler.fit_transform(raw.astype(np.float64))
    if save_scaler:
        np.save(C.RESULTS / 'scaler_mean.npy', scaler.mean_)
        np.save(C.RESULTS / 'scaler_scale.npy', scaler.scale_)
    else:
        if not (np.array_equal(scaler.mean_, np.load(C.RESULTS / 'scaler_mean.npy'))
                and np.array_equal(scaler.scale_, np.load(C.RESULTS / 'scaler_scale.npy'))):
            raise ValueError('Saved scaler does not match current features')
    return x, meta


def plot_curves(rows):
    specs = [('silhouette', 'Silhouette (fixed 5,000-image subset)', True),
             ('ch', 'Calinski-Harabasz (all source train)', True),
             ('db', 'Davies-Bouldin (all source train)', True),
             ('stability', 'Initialization stability (pairwise ARI)', True),
             ('gmm_bic', 'GMM BIC (diagonal covariance)', False)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, has_std) in zip(axes.flat, specs):
        values = [r[key + '_mean'] if has_std else r[key] for r in rows]
        errors = [r[key + '_std'] for r in rows] if has_std else None
        ks = [r['K'] for r in rows]
        ax.errorbar(ks, values, yerr=errors, marker='o', capsize=3)
        ax.set(title=title, xlabel='K', xticks=ks)
        ax.grid(alpha=.25)
        single, a = plt.subplots(figsize=(7, 4.5))
        a.errorbar(ks, values, yerr=errors, marker='o', capsize=3)
        a.set(title=title, xlabel='K', xticks=ks)
        a.grid(alpha=.25)
        single.tight_layout()
        single.savefig(C.FIGURES / f'{key}_vs_k.png', dpi=160)
        plt.close(single)
    ax = axes.flat[-1]
    ax.plot([r['K'] for r in rows], [r['min_cluster_fraction'] for r in rows], 'o-')
    ax.set(title='Smallest cluster fraction (lowest-inertia run)', xlabel='K', xticks=C.K_VALUES)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(C.FIGURES / 'k_selection_overview.png', dpi=180)
    plt.close(fig)


def silhouette_profile(k, labels, samples):
    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = 0
    for cluster in range(k):
        values = np.sort(samples[labels == cluster])
        y = np.arange(bottom, bottom + len(values))
        ax.fill_betweenx(y, 0, values, alpha=.7)
        ax.text(-.08, bottom + len(values)/2, str(cluster))
        bottom += len(values) + 20
    ax.axvline(samples.mean(), color='red', ls='--', label=f'mean={samples.mean():.4f}')
    ax.set(title=f'K={k}: lowest-inertia run; fixed evaluation subset', xlabel='Silhouette', ylabel='Cluster samples')
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES / f'silhouette_profile_k{k}.png', dpi=160)
    plt.close(fig)


def main():
    C.ensure_dirs()
    start = time.monotonic()
    x, extraction = load_features(save_scaler=True)
    idx = np.sort(np.random.default_rng(C.SAMPLE_SEED).choice(len(x), min(C.SILHOUETTE_SAMPLE_SIZE, len(x)), replace=False))
    np.save(C.RESULTS / 'silhouette_sample_indices.npy', idx)
    print('Computing one shared Euclidean distance matrix for silhouette...', flush=True)
    distances = pairwise_distances(x[idx], metric='euclidean')
    np.fill_diagonal(distances, 0.)
    metadata = read_csv(C.FEATURES / 'metadata.csv')
    all_rows, runs, pairs, composition = [], [], [], []
    candidates = C.RESULTS / 'candidates'
    candidates.mkdir(exist_ok=True)
    protocol = dict(k_values=C.K_VALUES, seeds=C.SEEDS, n_init=C.N_INIT, max_iter=C.MAX_ITER,
                    kmeans_init='k-means++', kmeans_algorithm='lloyd', kmeans_tol=1e-4,
                    silhouette_sample_size=len(idx), sample_seed=C.SAMPLE_SEED,
                    silhouette_metric='euclidean in standardized 256D space',
                    stability='pairwise ARI over five full-training-data initializations; ddof=1',
                    gmm_covariance=C.GMM_COVARIANCE, gmm_n_init=C.GMM_N_INIT,
                    gmm_max_iter=C.GMM_MAX_ITER, gmm_seed=0, gmm_reg_covar=1e-6,
                    numpy_version=np.__version__, sklearn_version=sklearn.__version__,
                    extraction=extraction, candidate_selection='minimum inertia over five runs',
                    final_k=None, country_balancing=False, fit_split='source train only')
    save_json(C.RESULTS / 'selection_protocol.json', protocol)
    for k in C.K_VALUES:
        assignments, metrics, best = [], [], None
        for seed in C.SEEDS:
            model = KMeans(n_clusters=k, init='k-means++', n_init=C.N_INIT,
                           max_iter=C.MAX_ITER, random_state=seed, algorithm='lloyd').fit(x)
            labels = model.labels_
            if len(np.unique(labels[idx])) != k:
                raise RuntimeError('Evaluation subset misses a cluster; enlarge the same subset for all K')
            sil = silhouette_samples(distances, labels[idx], metric='precomputed')
            row = dict(K=k, seed=seed, silhouette=float(sil.mean()),
                       ch=float(calinski_harabasz_score(x, labels)), db=float(davies_bouldin_score(x, labels)),
                       inertia=float(model.inertia_), iterations=int(model.n_iter_),
                       occupancy=json.dumps(np.bincount(labels, minlength=k).tolist()))
            runs.append(row)
            metrics.append(row)
            assignments.append(labels)
            if best is None or model.inertia_ < best[0].inertia_:
                best = (model, seed, sil)
            print(f'K={k} seed={seed} silhouette={row["silhouette"]:.5f} inertia={model.inertia_:.1f} elapsed={time.monotonic()-start:.1f}s', flush=True)
        ari = []
        for a, b in itertools.combinations(range(len(assignments)), 2):
            value = float(adjusted_rand_score(assignments[a], assignments[b]))
            ari.append(value)
            pairs.append(dict(K=k, seed_a=C.SEEDS[a], seed_b=C.SEEDS[b], ARI=value))
        model, seed, sil = best
        np.savez_compressed(candidates / f'k{k}.npz', labels=model.labels_, centers=model.cluster_centers_, seed=seed, inertia=model.inertia_)
        silhouette_profile(k, model.labels_[idx], sil)
        print(f'K={k}: fitting diagonal GMM, n_init={C.GMM_N_INIT}...', flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            gmm = GaussianMixture(n_components=k, covariance_type=C.GMM_COVARIANCE,
                                  n_init=C.GMM_N_INIT, max_iter=C.GMM_MAX_ITER, random_state=0,
                                  reg_covar=1e-6).fit(x)
        counts = np.bincount(model.labels_, minlength=k)
        result = dict(K=k)
        for key in ('silhouette', 'ch', 'db'):
            vals = [r[key] for r in metrics]
            result[key+'_mean'] = float(np.mean(vals))
            result[key+'_std'] = float(np.std(vals, ddof=1))
        result.update(stability_mean=float(np.mean(ari)), stability_std=float(np.std(ari, ddof=1)),
                      gmm_bic=float(gmm.bic(x)), gmm_converged=bool(gmm.converged_),
                      gmm_iterations=int(gmm.n_iter_), gmm_warnings='; '.join(str(w.message) for w in caught),
                      candidate_seed=seed, candidate_inertia=float(model.inertia_),
                      min_cluster_fraction=float(counts.min()/len(x)), cluster_occupancy=json.dumps(counts.tolist()))
        all_rows.append(result)
        for cluster in range(k):
            for country in C.COUNTRIES:
                n = sum(int(label == cluster and r['country'] == country) for label, r in zip(model.labels_, metadata))
                composition.append(dict(K=k, cluster=cluster, country=country, count=n,
                                        fraction_within_cluster=n/int(counts[cluster])))
        write_csv(C.RESULTS / 'k_selection_metrics.csv', all_rows)
        write_csv(C.RESULTS / 'k_selection_runs.csv', runs)
        write_csv(C.RESULTS / 'stability_pairs.csv', pairs)
        write_csv(C.RESULTS / 'candidate_country_composition.csv', composition)
        print(f'K={k} complete: ARI={result["stability_mean"]:.4f}, BIC={result["gmm_bic"]:.1f}, occupancy={counts.tolist()}', flush=True)
    plot_curves(all_rows)
    save_json(C.RESULTS / 'selection_complete.json', dict(elapsed_seconds=time.monotonic()-start,
              metrics_sha256=sha256(C.RESULTS / 'k_selection_metrics.csv'), final_k=None,
              status='awaiting human review; no final domains or training generated'))
    print('K comparison complete. FINAL_K remains unset.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
