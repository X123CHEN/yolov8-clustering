"""Inspect a candidate, or produce final domains only after config.FINAL_K is confirmed."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from threadpoolctl import threadpool_limits

import config as C
from common import read_csv, write_csv, save_json
from select_k import load_features


def analyze(k, candidate=False):
    if k not in C.K_VALUES:
        raise ValueError('K must have been evaluated in the frozen comparison')
    if not (C.RESULTS / 'selection_complete.json').exists():
        raise RuntimeError('Complete all candidate comparisons first')
    x, _ = load_features()
    metadata = read_csv(C.FEATURES / 'metadata.csv')
    saved = np.load(C.RESULTS / 'candidates' / f'k{k}.npz', allow_pickle=False)
    seed = int(saved['seed'])
    if candidate:
        labels, centers = saved['labels'], saved['centers']
        results = C.RESULTS / 'candidate_analysis' / f'k{k}'
        figures = C.FIGURES / 'candidate_analysis' / f'k{k}'
    else:
        model = KMeans(n_clusters=k, random_state=seed, n_init=C.N_INIT,
                       max_iter=C.MAX_ITER, algorithm='lloyd').fit(x)
        labels, centers = model.labels_, model.cluster_centers_
        if not np.array_equal(labels, saved['labels']):
            raise RuntimeError('Final refit differs from the reviewed candidate; review before publishing labels')
        results, figures = C.RESULTS, C.FIGURES
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    distance_sq = ((x-centers[labels])**2).sum(axis=1)
    np.save(results / 'cluster_centers.npy', centers)
    write_csv(results / ('candidate_labels.csv' if candidate else 'final_cluster_labels.csv'),
              [dict(**r, latent_domain=int(label), distance_sq=float(d)) for r,label,d in zip(metadata,labels,distance_sq)])
    counts = np.bincount(labels, minlength=k)
    write_csv(results / 'cluster_occupancy.csv',
              [dict(cluster=i, count=int(n), fraction=float(n/len(x))) for i,n in enumerate(counts)])
    country_matrix = np.zeros((len(C.COUNTRIES),k),dtype=int)
    raw_damage, valid_damage = np.zeros((k,4),dtype=int), np.zeros((k,4),dtype=int)
    for row, label in zip(metadata,labels):
        country_matrix[C.COUNTRIES.index(row['country']),label] += 1
        for b in json.loads(row['boxes_json']):
            raw_damage[label,b[4]] += 1
            if b[0]<b[2] and b[1]<b[3]:
                valid_damage[label,b[4]] += 1
    write_csv(results / 'cluster_composition.csv',
              [dict(country=c,cluster=j,count=int(country_matrix[i,j]),
                    fraction_within_country=float(country_matrix[i,j]/country_matrix[i].sum()),
                    fraction_within_cluster=float(country_matrix[i,j]/counts[j]))
               for i,c in enumerate(C.COUNTRIES) for j in range(k)])
    class_names = ('D00','D10','D20','D40')
    damage_rows=[]
    for j in range(k):
        for cls,name in enumerate(class_names):
            damage_rows.append(dict(cluster=j,damage_class=name,raw_instance_count=int(raw_damage[j,cls]),
                                    valid_instance_count=int(valid_damage[j,cls]),
                                    valid_instance_fraction=float(valid_damage[j,cls]/valid_damage[j].sum()) if valid_damage[j].sum() else None))
    write_csv(results / 'damage_distribution.csv',damage_rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,matrix,title in zip(axes,[country_matrix/country_matrix.sum(1,keepdims=True),country_matrix/country_matrix.sum(0,keepdims=True)],
                               ['Fraction within each country','Fraction within each cluster']):
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        for i in range(3):
            for j in range(k):
                ax.text(j,i,f'{matrix[i,j]:.2f}\n(n={country_matrix[i,j]})',ha='center',va='center',fontsize=8)
        ax.set(xticks=range(k),yticks=range(3),yticklabels=C.COUNTRIES,xlabel='Cluster',title=title)
        fig.colorbar(im,ax=ax)
    fig.tight_layout()
    fig.savefig(figures/'country_cluster_matrix.png',dpi=160)
    plt.close(fig)
    fractions=valid_damage/np.maximum(valid_damage.sum(1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(8,4))
    bottom=np.zeros(k)
    for cls,name in enumerate(class_names):
        ax.bar(range(k),fractions[:,cls],bottom=bottom,label=name)
        bottom+=fractions[:,cls]
    ax.set(xlabel='Cluster',ylabel='Fraction of valid damage instances',xticks=range(k),ylim=(0,1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures/'damage_distribution_by_cluster.png',dpi=160)
    plt.close(fig)
    pca=PCA(n_components=2,svd_solver='full')
    xy=pca.fit_transform(x)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    scatter=axes[0].scatter(xy[:,0],xy[:,1],c=labels,cmap='tab10',s=3,alpha=.5,vmin=-.5,vmax=9.5)
    axes[0].set_title(f'Candidate K={k}' if candidate else f'Final K={k}')
    for country in C.COUNTRIES:
        mask=np.array([r['country']==country for r in metadata])
        axes[1].scatter(xy[mask,0],xy[mask,1],s=3,alpha=.5,label=country)
    axes[1].legend(markerscale=3)
    axes[1].set_title('Original source country (same PCA coordinates)')
    for ax in axes:
        ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    fig.tight_layout()
    fig.savefig(figures/'pca_clusters.png',dpi=160)
    plt.close(fig)
    representatives=[]
    for cluster in range(k):
        members=np.flatnonzero(labels==cluster)
        nearest=members[np.argsort(distance_sq[members],kind='stable')[:9]]
        fig,axes=plt.subplots(3,3,figsize=(10,10))
        for ax in axes.flat:
            ax.axis('off')
        for rank,(ax,index) in enumerate(zip(axes.flat,nearest),1):
            row=metadata[index]
            with Image.open(row['image_path']) as im:
                im=im.convert('RGB')
                im.thumbnail((480,360))
                ax.imshow(im)
            ax.set_title(f'{row["image_id"]}\nd2={distance_sq[index]:.2f}',fontsize=8)
            representatives.append(dict(cluster=cluster,rank=rank,image_path=row['image_path'],distance_sq=float(distance_sq[index])))
        fig.suptitle(f'{"Candidate" if candidate else "Final"} K={k}, cluster {cluster}: nearest to center')
        fig.tight_layout()
        fig.savefig(figures/f'cluster_{cluster}_representatives.png',dpi=130)
        plt.close(fig)
    write_csv(results/'representatives.csv',representatives)
    idx=np.load(C.RESULTS/'silhouette_sample_indices.npy')
    save_json(results/'analysis_metadata.json',dict(K=k,seed=seed,candidate=candidate,
              silhouette=float(silhouette_score(x[idx],labels[idx])),ch=float(calinski_harabasz_score(x,labels)),
              db=float(davies_bouldin_score(x,labels)),pca_explained_variance=pca.explained_variance_ratio_.tolist(),
              center_space='standardized 256D; use results/scaler_mean.npy and scaler_scale.npy',
              damage_policy='Keep all images and raw annotations; valid-instance fractions exclude zero/negative-area boxes',
              excluded_degenerate_boxes=int((raw_damage-valid_damage).sum())))
    print(f'Analysis saved to {results}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=int,help='Preview only; does not create final domain labels')
    args=parser.parse_args()
    if args.candidate is None and C.FINAL_K is None:
        parser.error('FINAL_K is unset. Review results and obtain user confirmation before finalizing.')
    with threadpool_limits(limits=4):
        analyze(args.candidate if args.candidate is not None else C.FINAL_K,candidate=args.candidate is not None)
