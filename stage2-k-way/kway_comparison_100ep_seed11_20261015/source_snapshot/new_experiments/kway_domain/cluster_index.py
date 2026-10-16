"""Join fixed offline cluster IDs to the same source image records as stage one."""
import csv
from collections import Counter
from pathlib import Path

from utils.comparison_training import ROOT, prepare_data


def audit_source_labels():
    country = prepare_data('country_unique')
    latent = prepare_data('latent_unique')
    _, _, _, country_rows, country_records, country_ids, country_val, _ = country
    _, _, _, latent_rows, latent_records, latent_ids, latent_val, _ = latent
    if country_records != latent_records or country_rows != latent_rows or country_val != latent_val:
        raise ValueError('Country/latent Unique must use identical images, boxes and validation')
    artifact = ROOT / 'latent_domain_analysis/results/final_cluster_labels.csv'
    labels, composition = {}, Counter()
    with artifact.open() as stream:
        for row in csv.DictReader(stream):
            path = str(Path(row['image_path']).resolve(strict=True))
            if path in labels:
                raise ValueError(f'Duplicate image in fixed cluster artifact: {path}')
            if row['split'] != 'train' or row['country'] not in country[0]:
                raise ValueError('Cluster supervision must contain source training images only')
            label = int(row['latent_domain'])
            labels[path] = label
            composition[(row['country'], label)] += 1
    if labels != latent_ids or set(labels.values()) != {0, 1}:
        raise ValueError('Fixed cluster IDs do not match stage-one data records')
    return dict(images=len(labels), validation_images=len(country_val), coverage=1.,
                cluster_counts=[sum(c == k for c in labels.values()) for k in range(2)],
                country_cluster={f'{country}:{cluster}': count for (country, cluster), count in composition.items()},
                artifact=str(artifact), identical_unique_protocol=True)
